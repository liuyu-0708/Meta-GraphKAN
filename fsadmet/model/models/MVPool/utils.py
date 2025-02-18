import torch


import torch

def index_to_mask(index, size):
    # 创建一个全零的布尔类型的张量，其长度为给定的size。
    mask = torch.zeros((size, ), dtype=torch.bool)
    # 将指定索引的位置设置为1，形成掩码。
    mask[index] = 1
    return mask


# random_splits 函数接受一个图数据对象 data 和分类的数量 num_classes
# 这个函数是用于在图数据集上创建随机分割的实用工具
def random_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # * the rest for testing
    indices = []
    for i in range(num_classes):
        # 找到所有属于第i类的数据点的索引
        index = (data.y == i).nonzero().view(-1)
        # 随机排列这些索引
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data
