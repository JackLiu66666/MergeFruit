import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=None):
        super(DQN, self).__init__()
        
        # 默认隐藏层大小
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        # 构建网络层
        layers = []
        input_size = state_size
        
        # 输入层归一化
        layers.append(nn.LayerNorm(input_size))
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def create_dqn_model(state_size, action_size, hidden_sizes=None, device=None):
    """
    创建DQN模型
    
    Args:
        state_size (int): 状态空间大小
        action_size (int): 动作空间大小
        hidden_sizes (list, optional): 隐藏层大小列表. 默认 None
        device (torch.device, optional): 设备. 默认 None
    
    Returns:
        DQN: 构建好的DQN模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN(state_size, action_size, hidden_sizes)
    model.to(device)
    return model

def get_model_summary(model):
    """
    获取模型摘要信息
    
    Args:
        model (nn.Module): 模型实例
    
    Returns:
        str: 模型摘要
    """
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append("=" * 50)
    
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        summary.append(f"{name}: {param_count} parameters")
    
    summary.append("=" * 50)
    summary.append(f"Total parameters: {total_params}")
    
    return "\n".join(summary)

def print_model_architecture(model):
    """
    打印模型架构
    
    Args:
        model (nn.Module): 模型实例
    """
    print(get_model_summary(model))
    print("\nArchitecture:")
    print(model)

if __name__ == "__main__":
    # 示例用法
    state_size = 16  # 假设状态空间大小为16
    action_size = 4   # 假设动作空间大小为4
    
    # 创建默认模型
    model = create_dqn_model(state_size, action_size)
    print("Default model:")
    print_model_architecture(model)
    
