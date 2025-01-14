import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)

def tonumpy(img):
    return img.cpu().numpy().transpose(1,2,0)

def results2img(result_dic, pc_range):
    colors_plt = ['orange', 'b', 'g']
    boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
    scores_3d = result_dic['scores_3d']
    labels_3d = result_dic['labels_3d']
    pts_3d = result_dic['pts_3d']
    keep = scores_3d > 0.0

    fig = plt.figure(figsize=(2, 4))
    canvas = FigureCanvas(fig)  # Create a canvas for the figure
    ax = fig.add_subplot(111)
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')

    for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep], labels_3d[keep], pts_3d[keep]):

        pred_pts_3d = pred_pts_3d.numpy()
        pts_x = pred_pts_3d[:, 0]
        pts_y = pred_pts_3d[:, 1]
        ax.plot(pts_x, pts_y, color=colors_plt[pred_label_3d], linewidth=1, alpha=0.8, zorder=-1)
        ax.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d], s=1, alpha=0.8, zorder=-1)

        pred_bbox_3d = pred_bbox_3d.numpy()
        xy = (pred_bbox_3d[0], pred_bbox_3d[1])
        width = pred_bbox_3d[2] - pred_bbox_3d[0]
        height = pred_bbox_3d[3] - pred_bbox_3d[1]
        pred_score_3d = float(pred_score_3d)
        pred_score_3d = round(pred_score_3d, 2)
        s = str(pred_score_3d)

    # render the figure to a NumPy array
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  #H x W x 3 array
    plt.close(fig)

    return image/255.0

def pred_img(img, aerialimg, results, pc_range):
    '''TODO: 1.) check rgb/bgr color orders and range [0-255/0-1]
             2.) only 1 sample from a batch is currently logged
             3.) resize cam imgs to save memory
             4.) train cam imgs have random distortions as data augmentation,
             can this be reversed? '''

    # unnormalize the camera images
    norm = transforms.Normalize(mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375],
                            std=[1/58.395, 1/57.12, 1/57.375])
    batch,n,c,h,w = img.shape

    for b in range(batch):
        # stack the cam images //todo: camera images are not ordered correctly
        imgs = [tonumpy(norm(img[b][i])) for i in range(n)]
        top = np.hstack(imgs[:3])
        bot = np.hstack(imgs[3:])
        cams = np.vstack((top, bot))

        # stack aerial image and aerial map
        aerial = tonumpy(aerialimg[b])
        if aerial.shape[-1] == 6:
            aerial = np.hstack((aerial[:,:,:3], aerial[:,:,3:6]))
        
        aerial = resize(aerial, cams)
        input_imgs = np.hstack((cams, aerial))
        input_imgs /= 255.0 #tb expects [0-1 range]

        # prediction results
        res = results2img(results[b]['pts_bbox'], pc_range)

        #gt maybe?

        ret = np.hstack((input_imgs, resize(res, input_imgs)))

    return ret