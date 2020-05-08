# walk over all the images and read all image files
# fragile implementation
# if one image is deleted from either of the image, 
# system would have wrong correspondence
# not advised to use this 

for subdir, dirs, files in os.walk(self.rootdir):
    if "image_02/data" in subdir: #Left RGB Folder
        for file in files:
            if ".png" in file or ".jpg" in file:
                # if self.lefimages>500:
                #     break

                left_file = os.path.join(subdir, file)
                left_img = cv2.imread(left_file)
                if left.shape[0] > 1 and left.shape[1]>1:
                    self.left_images.append(left_file)
                    self.lefimages+=1
        
    if "image_03/data" in subdir: #Right RGB Folder
        for file in files:
            if ".png" in file or ".jpg" in file:
                # if self.rigimages>500:
                #     break
                right_file = os.path.join(subdir, file)

                self.right_images.append(right_file)
                self.rigimages+=1