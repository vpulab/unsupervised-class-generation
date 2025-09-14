from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 
#Dataset path
dataset_path = 'dataset_trucks/'

#All operations will be performed on the masks
masks_folder = 'ss/'
rgb_folder = 'rgb/'
mask_th = 0.4
gradient_th = 0.01

count = 0
with open(f'{dataset_path}/valid_indices_2.txt','w') as file:
    print("")




first_amount = 0
second_best = 0
second_best_n = 0
sample = 1088
lista = [sample]
lista.extend(range(sample))
lista.extend(range(sample+1,4000))


theta = 7#  #Lets assume classes have a "squared-like" shape or at least that it may be contained in it. As such for a given area K wich is img_arr.sum(), we get a max perimeter of img_arr.sum()    
def get_smoothness_ratio(mask):
    """
    Calculates smoothness as ratio of original to smoothed perimeter
    Returns: smoothness ratio (higher value indicates more irregular boundary)
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get original contour and perimeter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    
    original_contour = max(contours, key=cv2.contourArea)
    original_perimeter = cv2.arcLength(original_contour, closed=True)
    
    # Smooth the mask using morphological operations
    kernel = np.ones((3,3), np.uint8)
    smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    
    # Get smoothed contour and perimeter
    smoothed_contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(smoothed_contours) == 0:
        return 0
    
    smoothed_contour = max(smoothed_contours, key=cv2.contourArea)
    smoothed_perimeter = cv2.arcLength(smoothed_contour, closed=True)
    
    # Calculate ratio
    if smoothed_perimeter == 0:
        return 0
    
    smoothness_ratio = original_perimeter / smoothed_perimeter
    return smoothness_ratio

def get_contour_energy(mask):
    """
    Calculates contour energy based on boundary direction changes
    Returns: contour energy (higher value indicates more irregular boundary)
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    
    contour = max(contours, key=cv2.contourArea)
    
    # Convert contour to array of points
    points = contour.squeeze()
    
    # If contour is too small
    if len(points) < 3:
        return 0
    
    # Calculate angles between consecutive segments
    angles = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        p3 = points[(i + 2) % len(points)]
        
        # Vectors between points
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            continue
            
        cos_angle = dot_product / norms
        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    # Contour energy is the sum of absolute angle changes
    if len(angles) == 0:
        return 0
        
    contour_energy = np.sum(np.abs(np.diff(angles)))
    return contour_energy


def get_boundary_pixels(mask):
    # Convert to uint8 if not already in correct format
    if mask.dtype != np.uint8:
        # If the mask has values between 0 and 1
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Print some debugging info
    # print(f"Mask shape: {mask.shape}")
    # print(f"Mask min/max values: {mask.min()}, {mask.max()}")
    # print(f"Number of non-zero pixels: {np.count_nonzero(mask)}")
    
    # Method 1: Using morphological operations
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    boundary = mask - eroded
    
    # Method 2: Using contour finding
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours were found
    if len(contours) == 0:
        print("No contours found!")
        return boundary, 0
    
    # Print number of contours found
    # print(f"Number of contours found: {len(contours)}")
    
    # Find the largest contour if there are multiple
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get perimeter length
    perimeter = cv2.arcLength(largest_contour, closed=True)
    # print(f"Perimeter length: {perimeter}")
    
    return boundary, perimeter

def analyze_mask(mask):
    smoothness = get_smoothness_ratio(mask)
    energy = get_contour_energy(mask)
    # print(f"Smoothness ratio: {smoothness:.3f}")
    # print(f"Contour energy: {energy:.3f}")
    return smoothness, energy

second_im = None
for k in range(4000):
    nonvalid = False
    gradients = False

    try:
        with Image.open(dataset_path+masks_folder+str(k)+'.png') as img:
            #First filter by threshold.
            img_arr = np.array(img).astype(float)      
            r = img_arr.mean()

            if r > mask_th:
                nonvalid = True

            boundary, perimeter = get_boundary_pixels(img_arr)
            area = img_arr.sum()

            
            smoothness, energy = analyze_mask(img_arr)

            #Now we use the relation 4pi*area/perimeter^2
            th_g =  4*math.pi*area/perimeter**2

            halfim = img_arr[img_arr.shape[0]//2 :,:].sum()
            if  halfim/area > 0.85:
                nonvalid = True

            if smoothness < 1 or energy > 50:
                nonvalid = True
            with Image.open(dataset_path+rgb_folder+str(k)+'.png') as rgb:
                fig,axes = plt.subplots(2,2,figsize=(10,10))

                axes[0][0].set_title('RGB')
                axes[0][0].imshow(rgb)
                axes[1][0].set_title('Mask Ratio:{} Area:{} Perimeter:{} Smoothness:{} Energy:{}'.format(r,area,perimeter,smoothness,energy))
                axes[1][0].imshow(img)


                fig.suptitle('Image {}.png, threshold:{}'.format(k,th_g), fontsize=16)                    
                plt.show()
                plt.close()


            if nonvalid:
                count+=1
            else:
                with open(f'{dataset_path}/valid_indices_2.txt','a') as file:
                    file.write("{}.png \n".format(k))
    except:
        continue


print("Discarded {} images".format(count))
