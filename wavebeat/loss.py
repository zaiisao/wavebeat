import torch

class GlobalMSELoss(torch.nn.Module):
    def __init__(self):
        super(GlobalMSELoss, self).__init__()

    def forward(self, input, target):
        
        # beat errors
        target_beats = target[...,target == 1]
        input_beats = input[...,target == 1]

        beat_loss = torch.nn.functional.mse_loss(input_beats, target_beats)

        # no beat errors
        target_no_beats = target[...,target == 0]
        input_no_beats = input[...,target == 0]

        no_beat_loss = torch.nn.functional.mse_loss(target_no_beats, input_no_beats)

        return no_beat_loss + beat_loss, beat_loss, no_beat_loss

class GlobalBCELoss(torch.nn.Module):
    def __init__(self):
        super(GlobalBCELoss, self).__init__()

    def forward(self, input, target):
        
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]

        # beat errors
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]

        beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # no beat errors
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]

        downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # no downbeat errors
        target_no_downbeats = downbeat_act_target[downbeat_act_target == 0]
        input_no_downbeats = downbeat_act_input[downbeat_act_target == 0]

        no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # sum up losses
        total_loss = beat_loss + no_beat_loss + downbeat_loss + no_downbeat_loss

        return total_loss, beat_loss, no_beat_loss

class BCFELoss(torch.nn.Module):
    """ Binary cross-entropy false erorr. """
    def __init__(self):
        super(BCFELoss, self).__init__()

    def forward(self, input, target):
        
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]

        # beat errors
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]

        # The binary cross entropy with logits contains the sigmoid layer in it
        beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # no beat errors
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]

        downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # no downbeat errors
        target_no_downbeats = downbeat_act_target[downbeat_act_target == 0]
        input_no_downbeats = downbeat_act_input[downbeat_act_target == 0]

        no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # sum up losses
        total_beat_loss = 1/2 * ((beat_loss + no_beat_loss )**2 + (beat_loss - no_beat_loss)**2)
        total_downbeat_loss = 1/2 * ((downbeat_loss + no_downbeat_loss )**2 + (downbeat_loss - no_downbeat_loss)**2)

        # find form
        total_loss = total_beat_loss + total_downbeat_loss

        return total_loss, total_beat_loss, total_downbeat_loss
    
class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        # out is the output of the TCN without sigmoid and has two channels (two classifiers, beat objects and downbeat objects)
        # target is the binary tensor that represents beat and downbeat locations
        
        # split out the targets into beat targets and downbeat targets
        beat_target = target[:,0,:]
        downbeat_target = target[:,1,:]

        beat_out = out[:,0,:]
        downbeat_out = out[:,1,:]

        # beat errors
        target_beats = beat_target[beat_target == 1]
        input_beats =  beat_out[beat_target == 1]

        # The binary cross entropy with logits contains the sigmoid layer in it
        # beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # no beat errors
        target_no_beats = beat_target[beat_target == 0]
        input_no_beats = beat_out[beat_target == 0]

        # no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat errors
        target_downbeats = downbeat_target[downbeat_target == 1]
        input_downbeats = downbeat_out[downbeat_target == 1]

        # downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # no downbeat errors
        target_no_downbeats = downbeat_target[downbeat_target == 0]
        input_no_downbeats = downbeat_out[downbeat_target == 0]

        # no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # sum up losses
        # total_beat_loss = 1/2 * ((beat_loss + no_beat_loss )**2 + (beat_loss - no_beat_loss)**2)
        # total_downbeat_loss = 1/2 * ((downbeat_loss + no_downbeat_loss )**2 + (downbeat_loss - no_downbeat_loss)**2)

        # find form
        # total_loss = total_beat_loss + total_downbeat_loss

        # return total_loss, total_beat_loss, total_downbeat_loss

        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        # t represents the target beats for positive anchors
        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()
