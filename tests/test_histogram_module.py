width = 126

red_image = np.zeros((width,width,3), np.uint8)
green_image = np.zeros((width,width,3), np.uint8)
blue_image = np.zeros((width,width,3), np.uint8)
mix_image = np.zeros((width,width,3), np.uint8)

red_image[:,:] = (255,0,0)
green_image[:,:] = (0,255,0)
blue_image[:,:] = (0,0,255)
mix_image[:, 0:width//3] = (255,0,0)
mix_image[:, width//3:(width//3)*2] = (0,255,0)
mix_image[:, (width//3)*2:width] = (0,0,255)

num_bins = 5
red_hist, red_hist_3d = hist_module.rgb_hist(red_image.astype('double'), num_bins)
green_hist, green_hist_3d = hist_module.rgb_hist(green_image.astype('double'), num_bins)
blue_hist, blue_hist_3d = hist_module.rgb_hist(blue_image.astype('double'), num_bins)
mix_hist, mix_hist_3d = hist_module.rgb_hist(mix_image.astype('double'), num_bins)

red_hist_rg = hist_module.rg_hist(red_image.astype('double'), num_bins)
green_hist_rg = hist_module.rg_hist(green_image.astype('double'), num_bins)
blue_hist_rg = hist_module.rg_hist(blue_image.astype('double'), num_bins)
mix_hist_rg = hist_module.rg_hist(mix_image.astype('double'), num_bins)

plt.bar(np.array(range(1, red_hist.size + 1)), red_hist)
plt.bar(np.array(range(1, green_hist.size + 1)), green_hist)
plt.bar(np.array(range(1, blue_hist.size + 1)), blue_hist)
plt.bar(np.array(range(1, mix_hist.size + 1)), mix_hist)