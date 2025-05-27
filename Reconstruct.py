import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def reconstruct(filepath,num_iterations,t,neighbourhood,sigma=1):
	img = cv2.imread(filename=filepath)
	size_x = img.shape[1]
	size_y = img.shape[0]
	img[::2, ::2, :] = 0 
	img[1::2, 1::2, :] = 0
	cv2.imwrite("tree_blaced.jpg",img)
	zoom_img = np.copy(img)
	for i in range(num_iterations):
		print(i+1)
		der_x=cv2.Sobel(zoom_img,cv2.CV_64F,1,0,ksize=3)
		der_y=cv2.Sobel(zoom_img,cv2.CV_64F,0,1,ksize=3)
		der = np.concatenate((der_x[:,:,:,np.newaxis],der_y[:,:,:,np.newaxis]),axis=3) # h , w, 3, 2

		new_img = np.copy(zoom_img)

		for j in range(size_y-2*(neighbourhood//2)):
			for k in range(size_x-2*(neighbourhood//2)):
				if img[j+neighbourhood//2][k+neighbourhood//2][0]==0 and img[j+neighbourhood//2][k+neighbourhood//2][1]==0 and img[j+neighbourhood//2][k+neighbourhood//2][2]==0 :
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
					dum = np.sum(np.sum(zoom_img[j:j+neighbourhood,k:k+neighbourhood,:] * norm_kernel[:,:,np.newaxis],axis=1),axis=0)
					new_img[j+neighbourhood//2,k+neighbourhood//2,:] = dum
		zoom_img = new_img

	print(np.sum(np.sign(zoom_img-img)))
	cv2.imwrite(f"{filepath[:-4]}_new.png",zoom_img)
