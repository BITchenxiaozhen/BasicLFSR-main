from torch import nn
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import imageio
import time
import scipy
from scipy import io


def main(args):

    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size,pin_memory = True,shuffle=True)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)

    '''Caculate Parameter'''
    if args.param == True:
        params = list(net.parameters())
        num_params = sum([p.numel() for p in params])
        logger.log_string("The number of parameter is: %d" % num_params)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            # net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    net = net.to(device)

    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    # for p in optimizer.state.values():
    #     if isinstance(p, torch.Tensor):
    #         p.to(device)
    # optimizer = torch.optim.AdamW(
    #     [paras for paras in net.parameters() if paras.requires_grad == True],
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=args.decay_rate
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    '''Tensorboard'''
    if args.tensorboard:
        writer = SummaryWriter(log_dir='./tf_logs/'+args.model_name, filename_suffix=str(args.epoch), flush_secs=180)

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')

    all_iter = 0
    '''Test Best PSNR SSIM'''
    best_average_psnr = 0
    best_average_ssim = 0

    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))
        cur_iter = 0

        for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
            all_iter +=1
            cur_iter +=1

            [Lr_angRes_in, Lr_angRes_out] = data_info
            data_info[0] = Lr_angRes_in[0].item()
            data_info[1] = Lr_angRes_out[0].item()

            data = data.to(device, non_blocking=True)  # low resolution
            label = label.to(device, non_blocking=True)  # high resolution

            data_copy = data.clone().detach()
            out = net(data_copy, data_info)
            loss = criterion(out, label, data_info)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

            '''Tensorboard write log'''
            if cur_iter % args.iter == 0:
                test_average_psnr, test_average_ssim = test(args, test_Names, test_Loaders, net,writer, all_iter, logger)

                ''' Save Best PTH  '''
                if test_average_psnr > best_average_psnr:
                    best_average_psnr = test_average_psnr
                    save_pth(args, checkpoints_dir, logger, idx_epoch, net, best=True)
        "epoch val"
        test_average_psnr, test_average_ssim = test(args, test_Names, test_Loaders, net, writer, all_iter, logger)

        ''' Save PTH  '''
        if args.local_rank == 0 and test_average_psnr > best_average_psnr:
            best_average_psnr = test_average_psnr
            save_pth(args, checkpoints_dir, logger, idx_epoch, net, best=True)
        save_pth(args, checkpoints_dir, logger, idx_epoch, net)
        ''' scheduler '''
        scheduler.step()


def save_pth(args, checkpoints_dir, logger, idx_epoch, net, best = False):
    if args.local_rank == 0:
        temp = ''
        if best:
            temp += '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
        else:
            temp += '/best_model.pth'
        save_ckpt_path = str(checkpoints_dir) + temp
        state = {
            'epoch': idx_epoch + 1,
            'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
        }
        torch.save(state, save_ckpt_path)
        logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


def test(args, test_Names, test_loaders, net, writer,all_iter, logger):
    # net = Net(args.angRes, args.upscale_factor, args.channels)
    # net.to(args.device)
    # cudnn.benchmark = True

    # if os.path.isfile(args.model_path):
    #     model = torch.load(args.model_path, map_location={'cuda:0': args.device})
    #     net.load_state_dict(model['state_dict'])
    # else:
    #     print("=> no model found at '{}'".format(args.load_model))

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(args, test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            logger.log_string(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (
                test_name, psnr_epoch_test, ssim_epoch_test))

            writer.add_scalar(test_name+'PSNR', psnr_epoch_test, all_iter)
            writer.add_scalar(test_name+'SSIM', ssim_epoch_test, all_iter)


        average_psnr = sum(psnr_testset) / len(psnr_testset)
        average_ssim = sum(ssim_testset) / len(ssim_testset)
        logger.log_string(time.ctime()[4:-5] + ' Average_Result: Average_PSNR---%.6f, Average_SSIM---%.6f' % (
            (average_psnr), (average_ssim)))
        writer.add_scalar('average_PSNR', average_psnr, all_iter)
        writer.add_scalar('average_SSIM', average_ssim, all_iter)
    return average_psnr, average_ssim


def inference(args, test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(args.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // args.angRes_in, vw // args.angRes_in
        subLFin = LFdivide(data, args.angRes_in, args.patch_size_for_test, args.stride_for_test)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(
                    0)  # patchsize 128 tmp (1,1,640,640)  patchsize 32 tmp (1,1,160,160)
                with torch.no_grad():
                    # torch.cuda.empty_cache()
                    out = net(tmp.to(args.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, args.angRes_in, args.patch_size_for_test * args.scale_factor, args.stride_for_test * args.scale_factor,
                            h0 * args.scale_factor, w0 * args.scale_factor)

        psnr, ssim = cal_metrics(label, outLF, args.angRes_in)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        # save_path = args.save_path + '/' + args.model_name + '/'
        #
        # isExists = os.path.exists(save_path + test_name)
        # if not (isExists):
        #     os.makedirs(save_path + test_name)
        #
        # io.savemat(save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
        #                  {'LF': outLF.numpy()})
        # pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test



if __name__ == '__main__':
    from option import args

    main(args)



# def val(device, net, test_Names, test_Loaders, val_dir, writer, logger, idx_epoch,cur_inter, all_inter):
#
#     with torch.no_grad():
#         ''' Create Excel for PSNR/SSIM '''
#         excel_file = ExcelFile()
#
#         psnr_testset = []
#         ssim_testset = []
#         for index, test_name in enumerate(test_Names):
#             test_loader = test_Loaders[index]
#
#             epoch_dir = val_dir.joinpath('VAL_epoch_%02d_iter_%02d' % (idx_epoch + 1, cur_inter))
#             epoch_dir.mkdir(exist_ok=True)
#             save_dir = epoch_dir.joinpath(test_name)
#             save_dir.mkdir(exist_ok=True)
#
#             start = time.time()
#
#             psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
#             excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)
#
#             end = time.time()
#             print("________________________cost_time:" + str(end - start))
#             psnr_epoch_test = float(np.array(psnr_iter_test).mean())
#             ssim_epoch_test = float(np.array(ssim_iter_test).mean())
#
#             psnr_testset.append(psnr_epoch_test)
#             ssim_testset.append(ssim_epoch_test)
#             logger.log_string('The %d epoch %d inter Test on %s, psnr/ssim is %.2f/%.3f' % (
#                 idx_epoch + 1, cur_inter, test_name, psnr_epoch_test, ssim_epoch_test))
#
#             writer.add_scalar(test_name+'PSNR', psnr_epoch_test, all_inter)
#             writer.add_scalar(test_name+'SSIM', ssim_epoch_test, all_inter)
#             pass
#         psnr_mean_test = float(np.array(psnr_testset).mean())
#         ssim_mean_test = float(np.array(ssim_testset).mean())
#         logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
#                           % (psnr_mean_test, ssim_mean_test))
#
#         excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
#
#         writer.add_scalar('average_PSNR', psnr_mean_test, all_inter)
#         writer.add_scalar('average_SSIM', ssim_mean_test, all_inter)
#         pass
#     return psnr_mean_test, ssim_mean_test


# def test(test_loader, device, net, save_dir=None):
#     LF_iter_test = []
#     psnr_iter_test = []
#     ssim_iter_test = []
#     for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
#         [Lr_angRes_in, Lr_angRes_out] = data_info
#         data_info[0] = Lr_angRes_in[0].item()
#         data_info[1] = Lr_angRes_out[0].item()
#
#         Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
#         Hr_SAI_y = Hr_SAI_y
#         Sr_SAI_cbcr = Sr_SAI_cbcr
#
#         ''' Crop LFs into Patches '''
#         subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
#         numU, numV, H, W = subLFin.size()
#         subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
#         subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
#                                args.angRes_in * args.patch_size_for_test * args.scale_factor)
#
#         ''' SR the Patches '''
#         for i in range(0, numU * numV, args.minibatch_for_test):
#             tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
#             with torch.no_grad():
#                 net.eval()
#                 torch.cuda.empty_cache()
#                 out = net(tmp.to(device), data_info)
#                 subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
#         subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)
#
#         ''' Restore the Patches to LFs '''
#         Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
#                               args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
#         Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')
#
#         ''' Calculate the PSNR & SSIM '''
#         psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
#         psnr_iter_test.append(psnr)
#         ssim_iter_test.append(ssim)
#         LF_iter_test.append(LF_name[0])
#
#
#         ''' Save RGB '''
#         if save_dir is not None:
#             save_dir_ = save_dir.joinpath(LF_name[0])
#             save_dir_.mkdir(exist_ok=True)
#             views_dir = save_dir_.joinpath('views')
#             views_dir.mkdir(exist_ok=True)
#             Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
#             Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
#             Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)
#
#             # save the SAI
#             # path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
#             # imageio.imwrite(path, Sr_SAI_rgb)
#             # save the center view
#             img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
#             path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
#             imageio.imwrite(path, img)
#             # save all views
#             for i in range(args.angRes_out):
#                 for j in range(args.angRes_out):
#                     img = Sr_4D_rgb[i, j, :, :, :]
#                     path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
#                     imageio.imwrite(path, img)
#                     pass
#                 pass
#             pass
#         pass
#
#     return psnr_iter_test, ssim_iter_test, LF_iter_test
