input_shape = (576, 1024)
output_2d_shape = (144, 256)
feat_downscale = 4

base_height=1.786
x_range = (3, 103)
y_range = (-12, 12)
z_range = 7
meter_per_pixel = 0.5

bev_x = int((x_range[1] - x_range[0]) / meter_per_pixel)
bev_y = int((y_range[1] - y_range[0]) / meter_per_pixel)
bev_shape = (bev_x, bev_y)