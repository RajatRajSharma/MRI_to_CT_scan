# Next plan for HMSS.net
- 2 type of data Augmentation required
- Will have updated tranformation function that will work on 2 images in pair like this
```
mri_image, ct_image = self.transform(mri_image, ct_image)
```
-  3 channel to 1 channel as MRI/CT is BW only
-  Will not use large train data (atmost 1000 pair) if all the data doesn't handle lot of cases.
- Will have uni center data
  
