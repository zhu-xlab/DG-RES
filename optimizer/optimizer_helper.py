from torch import optim


def get_optim_and_scheduler(model, network, epochs, lr, nesterov=False):
    params = model.parameters()
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(params, lr=5e-5)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_style(style_net, epochs, lr, nesterov=False, step_radio=0.8):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * step_radio)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_layer_joint(style_net, epochs, lr, train_all=None, nesterov=False):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * 1.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


# def get_optim_and_scheduler(model, network, epochs, lr, momentum=.9, weight_decay=.0005, \
#                                         nesterov=False, step_radio=0.8):
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, \
#                             weight_decay=weight_decay, nesterov=nesterov)
#     step_size = int(epochs * step_radio)
#     schedulers = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
#     return optimizer, scheduler
