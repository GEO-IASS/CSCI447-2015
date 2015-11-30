import webcolors

def get_color_name(requested_color):
	min_colors = {}
	for key, name in webcolors.css3_hex_to_names.items():
		if name in ['grey','red','green','yellow','blue','magenta','cyan','white']:
			r_c, g_c, b_c = webcolors.hex_to_rgb(key)
			rd = (r_c - requested_color[0]) ** 2
			gd = (g_c - requested_color[1]) ** 2
			bd = (b_c - requested_color[2]) ** 2
			min_colors[(rd + gd + bd)] = name
	return min_colors[min(min_colors.keys())]

def printall():
	for rgb in [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]:
		requested_color = rgb
		closest_name = get_color_name(requested_color)
		print("Name:", closest_name)

#printall()