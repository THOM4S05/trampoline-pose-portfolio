import numpy as np

def apply_affine(points, matrix):
    points = np.asarray(points,dtype=float)
    return np.column_stack([points,np.ones(len(points))]) @ np.asarray(matrix).T

def udp_affine(center, scale, input_size=(192,256)):
    """MMPose UDP pixel-endpoint convention: center maps to (size-1)/2."""
    sx,sy=(np.asarray(input_size,dtype=float)-1)/np.asarray(scale)
    cx,cy=center
    return np.array([[sx,0,sx*(scale[0]/2-cx)],[0,sy,sy*(scale[1]/2-cy)]])

def inverse_affine(matrix):
    return np.linalg.inv(np.vstack([matrix,[0,0,1]]))[:2]

def crop_geometry(box,padding=1.25,input_size=(192,256)):
    box=np.asarray(box,dtype=float)
    center=(box[:2]+box[2:])/2
    scale=(box[2:]-box[:2])*padding
    ratio=input_size[0]/input_size[1]
    if scale[0] > scale[1]*ratio:
        scale[1]=scale[0]/ratio
    else:
        scale[0]=scale[1]*ratio
    matrix=udp_affine(center,scale,input_size)
    return dict(center=center.tolist(),scale=scale.tolist(),input_size=list(input_size),
                source_to_input_affine=matrix.tolist(),input_to_source_affine=inverse_affine(matrix).tolist(),
                padded_bbox_xyxy=np.r_[center-scale/2,center+scale/2].tolist(),
                pixel_convention='UDP endpoints; native-source affine warp before model resizing')

def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    area_a=np.maximum(0,a[2:]-a[:2]).prod()
    area_b=np.maximum(0,b[2:]-b[:2]).prod()
    return float(inter/max(area_a+area_b-inter,1e-9))

def rotated_boxes_to_source(boxes,rotation,width,height):
    """Invert CCW quarter-turn boxes in continuous pixel-edge coordinates."""
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    if len(boxes)==0 or rotation==0:
        return boxes.copy()
    x1,y1,x2,y2=boxes.T
    if rotation==90:
        return np.column_stack([width-y2,x1,width-y1,x2])
    if rotation==180:
        return np.column_stack([width-x2,height-y2,width-x1,height-y1])
    if rotation==270:
        return np.column_stack([y1,height-x2,y2,height-x1])
    raise ValueError('Unsupported detector rotation')
