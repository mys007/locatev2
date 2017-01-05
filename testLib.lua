require 'cunn'
require 'image'


local patchdir, subf = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset', 'maps'
--local patchdir, subf = '/media/simonovm/Slow/datasets/Locate/panors', 'cyl'
local shadesrange = {4,21}
local opath = ''

function getPatchDir(d)
    return d and paths.concat(patchdir, d, subf) or patchdir
end

--------------------
--as in donkeyModapairs
function loadImagePair(path, nocrop)
    local cr = nocrop and '' or '_crop'
    local input1 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, 'photo'..cr..'.png')) --4 channels (alpha)
    local input2
    
    if opt.inputMode=='allshades' or opt.inputMode=='allshadesG' then
        local nCh = opt.inputMode=='allshades' and 3 or 1
        for i=shadesrange[1],shadesrange[2] do
            local im = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, string.format('panorama'..cr..'_%02d.png', i)), nCh)
            input2 = input2 or torch.Tensor((shadesrange[2]-shadesrange[1]+1)*nCh,im:size(im:dim()-1),im:size(im:dim()))
            input2:narrow(1, nCh*(i-shadesrange[1])+1, nCh):copy(im)
        end
    elseif opt.inputMode=='camnorm' then
        input2 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, 'normalsCamera'..cr..'.png'), 3)
    elseif opt.inputMode=='depth' then
        input2 = torch.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, 'distance'..cr..'.t7img.gz')):decompress()
        input2 = input2:view(1,input2:size(1), input2:size(2))
        input2 = torch.log(input2 + 1.1)        
    elseif opt.inputMode=='shadesnormG' then
        local nCh = 1
        for i=shadesrange[1],shadesrange[2] do
            local im = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, string.format('panorama'..cr..'_%02d.png', i)), nCh)
            input2 = input2 or torch.Tensor(3+(shadesrange[2]-shadesrange[1]+1)*nCh,im:size(im:dim()-1),im:size(im:dim()))
            input2:narrow(1, nCh*(i-shadesrange[1])+1, nCh):copy(im)
        end        
        input2:narrow(1, input2:size(1)-2, 3):copy( image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subf, 'normalsCamera'..cr..'.png'), 3) )
    else
        assert(false)
    end
       
    return input1, input2
end

function normalizePair(input1, input2, model)
    -- mean/std
    for i=1,3 do
        input1[i]:add(-model.meanstd.mean[i])
        input1[i]:div(model.meanstd.std[i]) 
    end
    for i=1,input2:size(1) do
        input2[i]:add(-model.meanstd.mean[i+3])
        input2[i]:div(model.meanstd.std[i+3])
    end
end

--------------------
function loadNetwork(path)
    cutorch.setDevice(1)
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    cutorch.manualSeed(1)
    
    local model = torch.load(path)
    model = model:cuda()
    
    --handle legacy
    if not model.inputMode then model.inputMode = 'allshadesG' end

    print('Loaded '..path)
    print(model)
    return model
end
