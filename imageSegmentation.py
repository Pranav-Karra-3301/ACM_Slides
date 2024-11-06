import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to be larger
    scale_factor = 1.5
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    
    # Reshape the image for clustering
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Set number of segments (k)
    k = 10
    max_iters = 20
    
    # Initialize random centers
    centers = pixels[np.random.choice(len(pixels), k, replace=False)]
    
    for iteration in range(max_iters):
        # Calculate distances to centers
        distances = np.sqrt(((pixels - centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centers
        new_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
        
        # Create segmented image with current centers
        segmented = np.uint8(new_centers)[labels].reshape(image.shape)
        
        # Create masks for visualization
        colored_masks = np.zeros_like(image)
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        
        for i in range(k):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[labels.reshape(image.shape[:2]) == i] = 255
            color_mask = np.zeros_like(image)
            color_mask[mask == 255] = colors[i % len(colors)]
            colored_masks = cv2.addWeighted(colored_masks, 1, color_mask, 0.5, 0)
            
        # Create display window
        display_img = np.hstack((image, segmented, colored_masks, 
                               cv2.addWeighted(image, 0.7, colored_masks, 0.3, 0)))
        
        # Add titles
        title_bar = np.zeros((50, display_img.shape[1], 3), dtype=np.uint8)
        titles = ['Original', f'Segmented (Iter {iteration+1})', 'Segments', 'Overlay']
        x_offset = 0
        for title in titles:
            width = display_img.shape[1] // 4
            cv2.putText(title_bar, title, (x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            x_offset += width
            
        # Combine title bar and images
        display_img = np.vstack((title_bar, display_img))
        
        # Show the window
        cv2.imshow('Image Segmentation Progress', display_img)
        
        # Wait for a short time and check for 'q' key to quit
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
            
        # Check convergence
        if np.all(new_centers == centers):
            break
            
        centers = new_centers
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "stadium.png"
    segment_image(image_path)
