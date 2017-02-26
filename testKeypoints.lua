require 'cunn'
local cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'
require 'cv.imgproc'
require 'cv.highgui'
require 'testLib'
require 'ClassifAP'
--require 'torchzlib'
require 'gnuplot'
local lapp = require 'pl.lapp'

opt = lapp[[
   -n        (default ".")      netpath
   -s        (default "kanzelwandbahn")          setname
   -q        (default "r@s")        keypoints definition in the query image 
   -o        (default 0)            whether to use orig image (1) or _crop image (0)
   --patchSize (default -1)     patchSize (if non-standard then adapted by SPP)
   --nocrop (default 0)     whether to use full panoramas
]]

local cr = opt.nocrop>0 and '' or '_crop'

------------------
local function findKeypoints(detector, image, mask)
    local keyPts = detector:detect{image=image, mask=mask}
    
    --ignore around borders
    local imageSize = {image:size()[2], image:size()[1]}
    keyPts = cv.KeyPointsFilter.runByImageBorder{keyPts, imageSize, opt.patchSize/2+1}
    
    local pts = torch.Tensor(keyPts.size, 2) --because writing to keyPts (keyPts.data[i].pt.x = -1e5) sometimes breaks the underlying data, wtf...
    for i=1,keyPts.size do
       pts[i][1] = keyPts.data[i].pt.x
       pts[i][2] = keyPts.data[i].pt.y
    end
    
    --nms: remove points too close together (keep those with stronger response)
    local th = 8 --px 
    local quality = torch.Tensor(keyPts.size)
    for i=1,quality:size(1) do quality[i] = keyPts.data[i].response end
    local q,index = quality:sort(true)

    local nremoved = 0
    for ii=1,quality:size(1) do
        for jj=1,ii-1 do
            local i,j = index[ii], index[jj]
            local d = (pts[i][1] - pts[j][1])^2 + (pts[i][2] - pts[j][1])^2
            if d < 8^2 then pts[i][1] = -1e5; nremoved = nremoved + 1; break end
        end
    end

    return pts, keyPts
end

------------------
local function extractKeypoints(pts, imageT, imid)
    imid = imid or 1
    local patches = torch.Tensor(pts:size(1), imageT:size(1), opt.patchSize, opt.patchSize)
    local coords = torch.Tensor(pts:size(1), 3)

    local idx = 1    
    for i=1,pts:size(1) do
        local x,y = math.ceil(pts[i][1] - opt.patchSize/2), math.ceil(pts[i][2] - opt.patchSize/2)
        if x>0 and y>0 then
            local p = imageT:narrow(2,y,opt.patchSize):narrow(3,x,opt.patchSize)
            coords[idx][1] = pts[i][1]; coords[idx][2] = pts[i][2]; coords[idx][3] = imid;
            patches[idx]:copy(p)
            idx = idx + 1
        end
    end 
    print('Skipped '..(pts:size(1)-idx+1)..'/'..pts:size(1))
    return {patches:narrow(1,1,idx-1), coords:narrow(1,1,idx-1)}
end

------------------
local function keypoints(model, detector, setname, real, imid)
    print('Keypoints: '..setname)
    local input1T, input2T = loadImagePair(setname, opt.nocrop>0)
    local mask = cv.erode{src=torch.gt(input1T[4],0), kernel=torch.Tensor(5,5):fill(1)}
    input1T = input1T:narrow(1,1,3)        
    normalizePair(input1T, input2T, model)    

    if real=='r' then
        local image1 = cv.imread{paths.concat(getPatchDir(setname), 'photo'..cr..'.png')}
        return extractKeypoints(findKeypoints(detector, image1, mask), input1T, imid)
    elseif real=='s' then
        local image2 = cv.imread{paths.concat(getPatchDir(setname), 'panorama'..cr..'_12.png')}
        return extractKeypoints(findKeypoints(detector, image2), input2T, imid)
    elseif real=='r@s' then
        local image2 = cv.imread{paths.concat(getPatchDir(setname), 'panorama'..cr..'_12.png')}
        return extractKeypoints(findKeypoints(detector, image2), input1T, imid)        
    end
end

------------------
function fullMatch(net, pa, pb, timed, bs)
  local npa = pa:size(1)
  local npb = pb:size(1)
  local S = bs or 64
  local M = pa:size(3)
  local nf = pa:size(2) + pb:size(2)
  local n = math.ceil(npb/S)*S
  local timer = torch.Timer()
  local elapsed = 0
  local nprocessed = 0

  local batch = torch.FloatTensor(S,nf,M,M);
  local batchGpu = torch.CudaTensor(S,nf,M,M);
  local scores = torch.FloatTensor(npa, n)

    for i=1,npa do
      if not timed then xlua.progress(i,npa) end
      for k=1,S do batch[{k,{1,pa:size(2)},{},{}}] = pa:select(1,i) end
      for j=1,n,S do
        for k=0,S-1 do batch[{k+1,{pa:size(2)+1,nf},{},{}}] = pb:select(1, math.min(j+k, npb)) end
        
        batchGpu:copy(batch)
        if timed then collectgarbage(); cutorch.synchronize(); timer:reset() end      
        local output = net:forward(batchGpu)
        if timed then cutorch.synchronize(); elapsed = elapsed + timer:time().real end        
        
        if output:numel() > batchGpu:size(1) then
            scores[{i,{j,j+S-1}}]:copy(output:select(2,1))
        else
            scores[{i,{j,j+S-1}}]:copy(output)
        end
        nprocessed = nprocessed + S
      end
    end

  local vals, idxs = scores:max(2)
  idxs = idxs:squeeze()
  local matches = {}
  local m = -math.huge --vals:mean()
  for i=1,idxs:size(1) do
    if vals[i][1] > m then table.insert(matches, {i, idxs[i]}) end
  end
  return torch.IntTensor(matches), scores[{{},{1,npb}}], elapsed, nprocessed
end

------------------
local function plotTopKeypoints(ap, kps, image, k, matches)
    local energy = torch.Tensor(ap.scores)
    local correct = torch.Tensor(ap.correct)
    local threshold,index = energy:sort(true)
    
    local n = threshold:numel()
    local num_correct = 0
    for i = 1,n do
        local idx = index[i]  
        local kp
        if matches then kp = kps[matches[idx][2]] else kp = kps[idx] end
   
        if kp[3]==1 then
            num_correct = num_correct + correct[idx]
            local clr = {}
            if num_correct < k then
                clr = correct[idx]==1 and {0,255,0} or {0,0,255}
            else
                clr = correct[idx]==1 and {32,96,32} or {32,32,128}
            end
            cv.circle{image, {kp[1], kp[2]}, 4 + (correct[idx]==1 and 1 or 0), clr}
        end
    end
    --cv.imshow{"win1", image}
    --cv.waitKey{0}    
    return image
end

------------------
local function plotKeypointsFiltered(kps, image)
    for i = 1,kps:size(1) do
        local kp = kps[i]
        if kp[3]==1 then
            cv.circle{image, {kp[1], kp[2]}, 3, {0,255,0}}
        else
            cv.circle{image, {kp[1], kp[2]}, 3, {255,255,0}}
        end
    end
    return image
end

------------------
local function plotKeypoints(detector, img, title)
    local _, keyPts = findKeypoints(detector, img)

    -- show keypoints to the user
    local imgWithAllKeypoints = cv.drawKeypoints{img, keyPts}
    cv.setWindowTitle{title, keyPts.size .. " keypoints"}
    cv.imshow{title, imgWithAllKeypoints}
    --cv.waitKey{0}
end

------------------
------------------

local model = loadNetwork(paths.concat(opt.n, 'network.net'))
opt.inputMode = model.inputMode
opt.patchSize = model.inputDim[3]



local AGAST = cv.AgastFeatureDetector{threshold=40}--34}
local mser = cv.MSER{}--_min_area=250}  --bad, not along contures (real); probably because it's greyscale?
local fast = cv.FastFeatureDetector{threshold=25} --ok but must be more filtered (nms)
local kaze = cv.KAZE{threshold=1e-4, nOctaves=1}
local orb = cv.ORB{nfeatures=300,edgeThreshold=100} --doesn't like weak horizont

local desc = kaze

--[[
cv.namedWindow{"win1"}; cv.namedWindow{"win2"}
plotKeypoints(desc, cv.imread{paths.concat(getPatchDir(), opt.s, 'cyl', 'photo_crop.png')}, 'win1')
plotKeypoints(desc, cv.imread{paths.concat(getPatchDir(), opt.s, 'cyl', 'panorama_crop_12.png')}, 'win2')
cv.waitKey{0}
boom()--]]


--- Extract keypoints
local kpQ = keypoints(model, desc, opt.s, opt.q)
local kpDB = keypoints(model, desc, opt.s, 's')
local decoys = {}
--local decoys = {'kanzelwandbahn', 'finsteraarhorn05', 'sp_toedi2', 'bietschhorn', 'festijoch'}
for k,v in pairs(decoys) do
    if v~=opt.s then
        local kp = keypoints(model, desc, v, 's', k+1)
        kpDB[1] = torch.cat(kpDB[1], kp[1], 1)
        kpDB[2] = torch.cat(kpDB[2], kp[2], 1)
    end
end    

--[[
local image1 = cv.imread{paths.concat(getPatchDir(opt.s), 'photo'..cr..'.png')} 
image1 = plotKeypointsFiltered(kpQ[2], image1)
local image2 = cv.imread{paths.concat(getPatchDir(opt.s), 'panorama'..cr..'_12.png')} 
image2 = plotKeypointsFiltered(kpDB[2], image2)
cv.imshow{"win1", image1}; cv.imshow{"win2", image2}; cv.waitKey{0}    --]]

--- Compute matches
local matches, scores = fullMatch(model, kpQ[1], kpDB[1], false, 256)

--- Compute matching scores
local thres = 15 --px
local apFull = ClassifAP()
local apMatch = ClassifAP()

for i=1,scores:size(1) do
    for j=1,scores:size(2) do
        local good = false
        if kpQ[2][i][3] == kpDB[2][j][3] then
            local dist = (kpQ[2][i] - kpDB[2][j]):norm()
            if dist < thres then good = true end
        end
        apFull:add(good, scores[i][j])
    end
end

for i=1,scores:size(1) do
    local j = matches[i][2]
    local good = false
    if kpQ[2][i][3] == kpDB[2][j][3] then
        local dist = (kpQ[2][i] - kpDB[2][j]):norm()
        if dist < thres then good = true end
    end
    apMatch:add(good, scores[i][j])
end

local topk = 20
local apF, recall, precision = apFull:compute(topk)
print('Full AP/R/P', apF)--, recall, precision)
local apM, recall, precision = apMatch:compute(topk)
print('Match AP/R/P', apM)--, recall, precision)       
        
--- plot & save matching keypoints (until K positives are shown)        
local image1 = cv.imread{paths.concat(getPatchDir(opt.s), 'photo'..cr..'.png')} 
image1 = plotTopKeypoints(apMatch, kpQ[2], image1, topk)
local image2 = cv.imread{paths.concat(getPatchDir(opt.s), 'panorama'..cr..'_12.png')} 
image2 = plotTopKeypoints(apMatch, kpDB[2], image2, topk, matches)

local opath = paths.concat(opt.n, 'keypoints_ep'..model.epoch)
paths.mkdir(opath)
local oname = 'Q'..opt.s..'_D'..#decoys..'_Q'..opt.q..'--ap'..apM..'ff'..apF
cv.imwrite{opath..'/'..oname..'.png', image1}
cv.imwrite{opath..'/'..oname..'S.png', image2}

local epsfile = opath..'/'..oname..'-prec.eps'
os.execute('rm -f "' .. epsfile .. '"')
local epsfig = gnuplot.epsfigure(epsfile)
local maxr = math.min(100, precision:numel())
gnuplot.plot({'precision top'..maxr,precision:narrow(1,1,maxr),'-'})
gnuplot.grid('on')
gnuplot.axis{0,maxr,0,1}
gnuplot.plotflush()
gnuplot.close(epsfig)
