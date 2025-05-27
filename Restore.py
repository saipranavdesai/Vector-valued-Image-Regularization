import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def restore(filepath,num_iterations,t,ratio,neighbourhood,sigma=1):
	img = cv2.imread(filename=filepath)
	zoom_img = np.copy(img)
	size_x = zoom_img.shape[1]
	size_y = zoom_img.shape[0]

	scale_factor = ratio
	downsampled_image = cv2.resize(img, (int(scale_factor*size_x), int(scale_factor*size_y)))
	cv2.imwrite("bheem_small.jpg",downsampled_image)
	zoom_img = cv2.resize(downsampled_image,(size_x,size_y),interpolation = cv2.INTER_NEAREST)
	cv2.imwrite("bheem_zoomed.jpg",zoom_img)

	for i in range(num_iterations):
		der_x=cv2.Sobel(zoom_img,cv2.CV_64F,1,0,ksize=3)
		der_y=cv2.Sobel(zoom_img,cv2.CV_64F,0,1,ksize=3)
		der = np.concatenate((der_x[:,:,:,np.newaxis],der_y[:,:,:,np.newaxis]),axis=3) # h , w, 3, 2
		new_img = np.copy(zoom_img)
		for j in range(size_y-2*(neighbourhood//2)):
			for k in range(size_x-2*(neighbourhood//2)):
				# print("loop:",i,j,k)
				G = der[j+neighbourhood//2][k+neighbourhood//2].T @ der[j+neighbourhood//2][k+neighbourhood//2]
				smoothed_matrix = gaussian_filter(G, sigma)
				vals,vecs = np.linalg.eig(smoothed_matrix)
				sort_indices = np.argsort(vals)
				sorted_vals = vals[sort_indices]
				sorted_vecs = vecs[:, sort_indices]
				T = ((sorted_vecs[:,0:1]@sorted_vecs[:,0:1].T)/np.sqrt(1+np.sum(sorted_vals))) + ((sorted_vecs[:,1:2]@sorted_vecs[:,1:2].T)/(1+np.sum(sorted_vals)))
				T_inv = np.linalg.inv(T)
				kernel = np.zeros((neighbourhood,neighbourhood))
				for i1 in range(neighbourhood):
					for j1 in range(neighbourhood):
						x = np.array([i1-neighbourhood//2,j1-neighbourhood//2])
						kernel[i1][j1] = x.T @ T_inv @ x
				kernel = np.exp(-0.25 * kernel * (1/t))
				norm_kernel = kernel/np.sum(kernel)
				dum = np.sum(np.sum(zoom_img[j:j+5,k:k+5,:] * norm_kernel[:,:,np.newaxis],axis=1),axis=0)
				new_img[j+neighbourhood//2,k+neighbourhood//2,:] = dum
		zoom_img = new_img
		# print("Iteration",i,"completed")
	print(np.sum(np.sign(zoom_img-img)))
	cv2.imwrite(f"{filepath[:-4]}_restored.png",zoom_img)
