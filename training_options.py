"""Configurable training options; old author configs keep their defaults."""


def resolve_seed(training_config, override=None):
    if override is not None:
        return override
    configured = training_config.get('seed', '').strip()
    return int(configured) if configured else None


def build_scheduler(optimizer, training_config):
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    name = training_config.get('lr_scheduler', 'StepLR')
    gamma = float(training_config.get('lr_decay_rate', '0.9'))
    if name == 'StepLR':
        return StepLR(optimizer, step_size=int(training_config['decay_epoch']), gamma=gamma)
    if name == 'MultiStepLR':
        milestones = [int(value.strip()) for value in training_config['milestones'].split(',')]
        if not milestones or any(value <= 0 for value in milestones):
            raise ValueError('milestones must be positive epoch counts')
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    raise ValueError(f'Unknown lr_scheduler: {name}')
