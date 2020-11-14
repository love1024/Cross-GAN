from PIL import Image
import util
from InGAN import InGAN
from configs import Config
from traceback import print_exc
import numpy as np
import torch


def test_one_scale(gan, input_tensor, scale, must_divide, affine=None, return_tensor=False, size_instead_scale=False):
    with torch.no_grad():
        in_size = input_tensor.shape[2:]
        if size_instead_scale:
            out_size = scale
        else:
            out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                        np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))

        output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                       input_size=in_size,
                                       output_size=out_size,
                                       rand_affine=affine,
                                       run_d_pred=False,
                                       run_reconstruct=False)
        if return_tensor:
            return output_tensor[1]
        else:
            return util.tensor2im(output_tensor[1])


def concat_images(images, margin, input_spot):
    h_sizes = [im.shape[0] for im in list(zip(*images))[0]]
    w_sizes = [im.shape[1] for im in images[0]]
    h_total_size = np.sum(h_sizes) + margin * (len(images) - 1)
    w_total_size = np.sum(w_sizes) + margin * (len(images) - 1)

    collage = np.ones([h_total_size, w_total_size, 3]) * 255
    for i in range(len(images)):
        for j in range(len(images)):
            top_left_corner_h = int(np.sum(h_sizes[:j]) + j * margin)
            top_left_corner_w = int(np.sum(w_sizes[:i]) + i * margin)
            bottom_right_corner_h = int(top_left_corner_h + h_sizes[j])
            bottom_right_corner_w = int(top_left_corner_w + w_sizes[i])

            if [i, j] == input_spot:
                collage[top_left_corner_h - margin//2: bottom_right_corner_h + margin//2,
                        top_left_corner_w - margin//2: bottom_right_corner_w + margin//2,
                        :] = [255, 0, 0]
            collage[top_left_corner_h:bottom_right_corner_h, top_left_corner_w:bottom_right_corner_w] = images[j][i]

    return collage


def generate_images_for_collage(gan, input_tensor, scales, must_divide):
    # NOTE: scales here is different from in the other funcs: here we only need 1d scales.
    # Prepare output images list
    output_images = [[[None] for _ in range(len(scales))] for _ in range(len(scales))]

    # Run over all scales and test the network for each one
    for i, scale_h in enumerate(scales):
        for j, scale_w in enumerate(scales):
            output_images[i][j] = test_one_scale(gan, input_tensor, [scale_h, scale_w], must_divide)
    return output_images

def generate_collage_and_outputs(conf, gan, input_tensor):
    output_images = generate_images_for_collage(gan, input_tensor, conf.collage_scales, conf.must_divide)

    for i in range(len(output_images)):
        for j in range(len(output_images)):
            Image.fromarray(output_images[i][j], 'RGB').save(conf.output_dir_path + '/test_%d_%d.png' % (i, j))

    input_spot = conf.collage_input_spot
    output_images[input_spot[0]][input_spot[1]] = util.tensor2im(input_tensor)

    collage = concat_images(output_images, margin=10, input_spot=input_spot)

    Image.fromarray(np.uint8(collage), 'RGB').save(conf.output_dir_path + '/test_collage.png')

def main():
    conf = Config().parse(create_dir_flag=False)
    conf.name = 'TEST_' + conf.name
    conf.output_dir_path = util.prepare_result_dir(conf)
    gan = InGAN(conf)

    try:
        gan.resume(conf.test_params_path, test_flag=True)
        [input_tensor] = util.read_data(conf)

        if conf.test_collage:
            generate_collage_and_outputs(conf, gan, input_tensor)

        print('Done with %s' % conf.input_image_path)

    except KeyboardInterrupt:
        raise
    except Exception as e:
        # print 'Something went wrong with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)
        print_exc()


if __name__ == '__main__':
    main()
