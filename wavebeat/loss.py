import torch

class GlobalMSELoss(torch.nn.Module):
    def __init__(self):
        super(GlobalMSELoss, self).__init__()

    def forward(self, input, target):
        
        # beat losses
        target_beats = target[...,target == 1]  # MJ: The ellipsis ... means as many : as possible.
#                                                 random_array = np.random.rand(2, 2, 2, 2)
#                                                 In such case, [:, :, :, 0] and [..., 0] are the same
        input_beats = input[...,target == 1]

        beat_loss = torch.nn.functional.mse_loss(input_beats, target_beats)

        # no beat losses
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

        # beat losses
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]

        beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # non-beat (background) losses
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]

        downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # non-downbeat (background) losses
        target_no_downbeats = downbeat_act_target[downbeat_act_target == 0]
        input_no_downbeats = downbeat_act_input[downbeat_act_target == 0]

        no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # sum up losses
        total_loss = beat_loss + no_beat_loss + downbeat_loss + no_downbeat_loss

        return total_loss, beat_loss, no_beat_loss

class BCFELoss(torch.nn.Module):
    """ Binary cross-entropy mean false erorr. """
    def __init__(self):
        super(BCFELoss, self).__init__()

    def forward(self, input, target):
        
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]

        # beat losses
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]

        # The binary cross entropy with logits contains the sigmoid layer in it
        beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # no beat losses
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat losses
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]

        downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # no downbeat losses2402
        
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
    #MJ: BCE: https://curt-park.github.io/2018-09-19/loss-cross-entropy/
    #(1) L(p∣x)에 대한 loglikelihood가 바로 negative binary cross entropy의 형태임을 알 수 있다.
    #  Binary cross entropy는 파라미터 π 를 따르는 베르누이분포와 관측데이터의 분포가 얼마나 다른지를 나타내며, 
    # 를 최소화하는 문제는 관측데이터에 가장 적합한(fitting) 베르누이분포의 파라미터 π를 추정하는 것으로 해석할 수 있다.
    # BCE1(y; p) = ∑_{i=1}^{n} [ y_{i}*log(p)+(1−y_{i})log(1−p)), 
    #   where y_{i} is a vector of target probabilities that audio sample points are beats, and 
    #    p is a vector of predicted probabilities that audio sample points are beats
    #  BCE1(y; p) is a vector of binary cross entropy loss.
     
    # BCE2(y; p) = ∑_{i=1}^{n} [ y_{i}*log(p)+(1−y_{i})log(1−p)), 
    #   where y_{i} is a vector of target probabilities that audio sample points are downbeats, and 
    #     p is a vector of predicted probabilities that audio sample points are downbeats
    
    # FL(y;p) = ∑_{i=1}^{n} [ alpha* (1 - p) ** gamma * y_{i}*log(p) + (1-alpha) * p ** gamma * (1−y_{i})log(1−p)) ]
    
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        # preds is the output of the TCN without sigmoid and has two channels (two classifiers, beat objects and downbeat objects)
        # targets is the binary tensor that represents beat and downbeat target locations
        
        # # split out the targets into beat targets and downbeat targets
        # beat_target_bxn = targets[:,0,:]
        # downbeat_target_bxn = targets[:,1,:]

        # beat_pred_bxn = preds[:,0,:]
        # downbeat_pred_bxn = preds[:,1,:]

        # # beat errors: extract value 1's from beat_target and value 1's from beat_out
        # target_beats = beat_target_bxn[beat_target_bxn == 1]
        # input_beats =  beat_pred_bxn[beat_target_bxn == 1]

        # # The binary cross entropy with logits contains the sigmoid layer in it
        # # beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # # no beat errors:  extract value 0's from beat_target and value 0's from beat_out
        # target_no_beats = beat_target_bxn[beat_target_bxn == 0]
        # input_no_beats = beat_pred_bxn[beat_target_bxn == 0]

        # # no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # # downbeat errors: extract value 1's from downbeat_target and value 1's from downbeat_out
        # target_downbeats = downbeat_target_bxn[downbeat_target_bxn == 1]
        # input_downbeats = downbeat_pred_bxn[downbeat_target_bxn == 1]

        # # downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # # no downbeat errors:  extract value 0's from downbeat_target and value 0's from downbeat_o
        # target_no_downbeats = downbeat_target_bxn[downbeat_target_bxn == 0]
        # input_no_downbeats = downbeat_pred_bxn[downbeat_target_bxn == 0]

        # # no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # # sum up losses:  total_beat_loss = beat_loss**2 + no_beat_loss **2, which is the same as
        # # total_beat_loss = 1/2 * ((beat_loss + no_beat_loss )**2 + (beat_loss - no_beat_loss)**2)
        # # total_downbeat_loss = 1/2 * ((downbeat_loss + no_downbeat_loss )**2 + (downbeat_loss - no_downbeat_loss)**2)

        # # find form
        # # total_loss = total_beat_loss + total_downbeat_loss

        # # return total_loss, total_beat_loss, total_downbeat_loss

        n_class = preds.shape[1]
        class_ids_1xc = torch.arange(
             0, n_class, dtype=targets.dtype, device=targets.device
         ).unsqueeze(0)

        # t represents the target beats for positive anchors
        #targets_bx1xcxn = targets.unsqueeze(1) #  targets: shape = (B,2,W) = (B,C,W)
        targets_bxcxn = targets
        p_bxcxn = torch.sigmoid(preds)  # shape of pred_mat = (B,C,W) = (B,2,W)
        p_bxcxn = torch.clamp(p_bxcxn, 1e-3, 1.0 - 1e-3)

        gamma = self.gamma
        alpha = self.alpha

        term1_bxcxn = (1 - p_bxcxn) ** gamma * torch.log(p_bxcxn)
        term2_bxcxn = p_bxcxn ** gamma * torch.log(1 - p_bxcxn)

        # print(term1.sum(), term2.sum())
        #MJ:  In the following, several broadcasting operations are performed. Verify if they work correct.  
        # bce_loss_bx1xcxn = (
        #     -(targets_bx1xcxn  == class_ids_1xc).float() * alpha * term1_bxcxn
        #     - (( targets_bx1xcxn  != class_ids_1xc) * (targets_bx1xcxn >= 0)).float() * (1 - alpha) * term2_bxcxn
        # )
        
        # bce_loss_bx1xcxn = (
        #     -(targets_bx1xcxn  == 1).float() * alpha * term1_bxcxn
        #     - ( targets_bx1xcxn  == 0).float() * (1 - alpha) * term2_bxcxn
        # )
        
        #MJ: targets_bxcxn conaints BxCxN samples, which may be positive or negative. 
        # (targets_bxcxn[:,0,:]  == 1) selects the positive samples for beat from each sample location in each audio in the batch
        #  whereas  ( targets_bxcxn  == 0) selects the negatives.
        # bce_loss_bxcxn[:,0:,:] is the bce of the beat classifier (for each audio sample point in each batch) 
        # and  bce_loss_bxcxn[:,1:,:] is the bce of the downbeat classifier. 
        #  p_bxcxn[i,0,k] is the probability that audio sample k in audio i in the batch is a beat. 
        # If the probability  approaches 1,  the sample k is more likely to be a beat. 
        # If the probability  approaches 0, the sample k is more likely to be background.
        # p_bxcxn[i,1,k] is the probability that audio sample k in audio i in the batch is a downbeat.
        bce_loss_bxcxn = (
            -(targets_bxcxn  == 1).float() * alpha * term1_bxcxn
            - ( targets_bxcxn  == 0).float() * (1 - alpha) * term2_bxcxn
        )
        

        # pos_ids = torch.nonzero(targets_bxcxn> 0)
        # return  bce_loss_bxcxn.sum()/ torch.clamp(pos_ids.numel(), min=1.0)
        
        return bce_loss_bxcxn.sum() / torch.clamp(targets_bxcxn.sum(), min=1.0)

    #MJ: torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats) computes the mean by default.
    # So, we compute the average bce focal loss.
    
    #MJ: Unstability of Sigmoid + BCE:
    # refer to https://stackoverflow.com/questions/69454806/sigmoid-vs-binary-cross-entropy-loss
    # BCEWithLogitsLoss (which is the same as F.binary_cross_entropy_with_logits): It is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer,
    # we take advantage of the log-sum-exp trick for numerical stability.
    # So, if you get some numerical problem with the current approach, you can use 
    #  BCEWithLogitsLoss (which is the same as F.binary_cross_entropy_with_logits)