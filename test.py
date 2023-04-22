import torch
a = torch.load('checkpoints/detr-r101-gref.pth')
torch.save(a,'checkpoints/detr-r101-gref_new.pth',_use_new_zipfile_serialization=False)
print ('asdf')