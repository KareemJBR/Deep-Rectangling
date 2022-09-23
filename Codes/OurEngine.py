def extract_mesh(_input_img, gt_img, _mask):
    # extract mesh
    input_img_mesh = _input_img[:, :, 0:3]
    gt_img_mesh = gt_img[:, :, 0:3]
    mask_mesh = _mask[:, :, 0:3]

    return input_img_mesh, gt_img_mesh, mask_mesh
