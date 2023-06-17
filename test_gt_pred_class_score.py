

import torch

batch_size = 3
n_anchors = 4
n_classes = 5
n_gt_objects = 6

pred_class_score = torch.rand([batch_size, n_anchors, n_classes])

gt_class_index = torch.randint(n_classes, [batch_size, n_gt_objects])

batch_index = torch.arange(batch_size).view(-1, 1).repeat(1, n_gt_objects)

gt_pred_class_score = pred_class_score[batch_index, :, gt_class_index]

print(gt_pred_class_score.shape == torch.Size([batch_size, n_gt_objects, n_anchors]))

# print(gt_pred_class_score)

for curr_batch_index in range(batch_size):
    for curr_gt_index in range(n_gt_objects):
        for curr_anchor_index in range(n_anchors):
            assert (pred_class_score[curr_batch_index, curr_anchor_index, gt_class_index[curr_batch_index, curr_gt_index]] == \
                    gt_pred_class_score[curr_batch_index, curr_gt_index, curr_anchor_index])
