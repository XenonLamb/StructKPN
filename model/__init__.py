import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['network']['whichModel']

    if model == 'KPN':
        from model.kpn import KPN
        m = KPN(opt['network'])
    elif model == 'KPN_S':
        from model.kpn import KPNSlim
        m = KPNSlim(opt['network'])
    elif model == 'RedCNN':
        from model.red_cnn import RedCNN, InitializeWith3mmWeight
        m = RedCNN()
        if opt['network']['initializeWith3mmWeights']:
            m = InitializeWith3mmWeight(m, opt['network'])
    elif model == 'KPN_RED':
        from model.kpn_red import KPN
        m = KPN(opt['network'])
    elif model == 'KPN_RED2':
        from model.kpn_red import KPN2
        m = KPN2(opt['network'])
    elif model == 'RedCNN_L':
        from model.red_cnn import RedCNNLARGE
        m = RedCNNLARGE()
    elif model == 'RedCNN_W':
        from model.red_cnn import RedCNNWIDE
        m = RedCNNWIDE()
        #if opt['network']['initializeWith3mmWeights']:
        #    m = InitializeWith3mmWeight(m, opt['network'])
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    """
       elif model == 'KPN_DIL':
           from model.kpn_dil import KPN
           m = KPN(opt['network'])
       elif model == 'KPN_UNET':
           from model.kpn_unet import KPN
           m = KPN(opt['network'])
       elif model == 'KPN_MEMNET':
           from model.kpn_memnet import KPN
           m = KPN(opt['network'])
       elif model == 'KPN_CARN':
           from model.kpn_carn import KPN
           m = KPN(opt['network'])
       """
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

if __name__ == '__main__':
    from kpn import KPN