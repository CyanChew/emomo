import torch

class ActionNormalizer:
    def __init__(self, action_min: torch.Tensor, action_max: torch.Tensor):
        assert action_min.dim() == 1 and action_min.size() == action_max.size()
        assert action_min.dtype == torch.float32
        assert action_max.dtype == torch.float32

        self.action_dim = action_min.size(0)
        self.min = action_min
        self.max = action_max

        # Store the range for each dimension
        self.range = self.max - self.min

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        self.range = self.range.to(device)

    def normalize(self, value: torch.Tensor):
        shape = value.size()
        
        # Ensure value is in the correct shape [B, Time, Action Dim]
        value = value.reshape(-1, self.action_dim)

        # Create a mask for dimensions where the range is zero
        zero_range_mask = self.range == 0
        
        # Apply normalization for non-zero ranges
        normed_value = (2 * (value - self.min) / self.range - 1)
        
        # Set normalized values for dimensions with zero range to 0 (or some constant)
        normed_value[:, zero_range_mask] = 0
        
        # Ensure the normalized value is clipped to [-1, 1]
        normed_value = torch.clamp(normed_value, min=-1.0, max=1.0)
        
        # Reshape back to original [B, Time, Action Dim]
        normed_value = normed_value.reshape(shape)
        
        return normed_value

    def denormalize(self, normed_value: torch.Tensor):
        shape = normed_value.size()
        
        # Ensure normed_value is in the correct shape [B, Time, Action Dim]
        normed_value = normed_value.reshape(-1, self.action_dim)

        # Reverse normalization for non-zero ranges
        value = (normed_value + 1) * self.range / 2 + self.min
        
        # Set denormalized values for dimensions with zero range to the constant value (min/max)
        value[:, self.range == 0] = self.min[self.range == 0]
        
        # Reshape back to original [B, Time, Action Dim]
        value = value.reshape(shape)
        
        return value


if __name__ == '__main__':
    # Test the action range and normalization to [-1, 1]
    action_min = torch.tensor([0.48582, -0.62086, 0.28755, -1, -1.32579, -1.14720, 0.00000, 1.00000], dtype=torch.float32)
    action_max = torch.tensor([0.58517, -0.06301, 0.35654, 1, -1.18260, 1.19256, 1.00000, 4.00000], dtype=torch.float32)
    
    normalizer = ActionNormalizer(action_min, action_max)
    
    # Sample tensor for normalization with shape [B, Time, Action Dim]
    sample_value = torch.tensor([[[0.5, -0.5, 0.3, 0.06, -1.25, 1.0, 1.0, 1.5]]], dtype=torch.float32)  # B=1, Time=1, Action Dim=8
    
    # Normalize and denormalize
    normalized_value = normalizer.normalize(sample_value)
    denormalized_value = normalizer.denormalize(normalized_value)
    print(sample_value)
    print(normalized_value)
    print(denormalized_value)
