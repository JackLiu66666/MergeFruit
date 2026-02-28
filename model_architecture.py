import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=None, use_dueling=False):
        super(DQN, self).__init__()
        self.use_dueling = use_dueling
        
        # 默认隐藏层大小
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        
        # 输入层归一化
        self.norm = nn.LayerNorm(state_size)
        
        # 构建共享特征提取层
        self.shared_layers = nn.ModuleList()
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(input_size, hidden_size))
            self.shared_layers.append(nn.ReLU())
            input_size = hidden_size
        
        if use_dueling:
            # Dueling DQN分支
            self.value_stream = nn.Linear(input_size, 1)
            self.advantage_stream = nn.Linear(input_size, action_size)
        else:
            # 传统DQN输出层
            self.output_layer = nn.Linear(input_size, action_size)
    
    def forward(self, x):
        x = self.norm(x)
        
        # 前向传播共享层
        for layer in self.shared_layers:
            x = layer(x)
        
        if self.use_dueling:
            # 计算状态价值和优势函数
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # 计算Q值：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # 传统DQN输出
            return self.output_layer(x)

def create_dqn_model(state_size, action_size, hidden_sizes=None, use_dueling=True, device=None):
    """
    创建DQN模型
    
    Args:
        state_size (int): 状态空间大小
        action_size (int): 动作空间大小
        hidden_sizes (list, optional): 隐藏层大小列表. 默认 None
        use_dueling (bool, optional): 是否使用Dueling DQN. 默认 True
        device (torch.device, optional): 设备. 默认 None
    
    Returns:
        DQN: 构建好的DQN模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN(state_size, action_size, hidden_sizes, use_dueling)
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
    state_size = 601  # 实际状态空间大小
    action_size = 20   # 实际动作空间大小（20个下落位置）
    
    # 创建Dueling DQN模型
    model = create_dqn_model(state_size, action_size, use_dueling=True)
    print("Dueling DQN model:")
    print_model_architecture(model)
    
    # 创建传统DQN模型进行对比
    traditional_model = create_dqn_model(state_size, action_size, use_dueling=False)
    print("\n" + "-" * 60 + "\n")
    print("Traditional DQN model:")
    print_model_architecture(traditional_model)
    
