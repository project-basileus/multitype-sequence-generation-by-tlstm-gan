	import mmd
	mmd = reload(mmd)
	from scipy.linalg import sqrtm
	
	def calculate_FID_batch(batch1, batch2, weight=1):
	    mu1, sigma1 = batch1.mean(axis=0), np.cov(batch1, rowvar=False)
	    mu2, sigma2 = batch2.mean(axis=0), np.cov(batch2, rowvar=False)
	    ssdiff = np.sum((mu1 - mu2)**2.0)
	    # calculate sqrt of product between cov
	    covmean = sqrtm(sigma1.dot(sigma2))
	    # check and correct imaginary numbers from sqrt
	    if np.iscomplexobj(covmean):
	        covmean = covmean.real
	    # calculate score
	    fid2 = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	    return fid2
	
	# test_et, test_ts is from positive training set
	X = discriminator([test_et, test_ts])[1]
	
	# g1_et, g1_ts is from generated data
  Y_2 = discriminator([g1_et, g1_ts])[1]
	
  fid = (calculate_FID_batch(X_pos.numpy(), Y_2.numpy()))
  mmd = (mmd.rbf_mmd2(X_pos, Y_2))
