from transformers import CLIPProcessor
from transformers.tokenization_utils_base import BatchEncoding
import torch
from transformers.image_processing_utils import BatchFeature

def preprocess(
    self,
    images: torch.Tensor,
    return_tensors: str = "pt"
) :
    do_resize = do_resize if do_resize is not None else self.do_resize
    size = size if size is not None else self.size
    size = get_size_dict(size, param_name="size", default_to_square=False)
    resample = resample if resample is not None else self.resample
    do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
    crop_size = crop_size if crop_size is not None else self.crop_size
    crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else self.do_normalize
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

    images = make_list_of_images(images)

    if not valid_images(images):
        raise ValueError(
            "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
            "torch.Tensor, tf.Tensor or jax.ndarray."
        )
    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_center_crop=do_center_crop,
        crop_size=crop_size,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )

    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]
    # All transformations expect numpy arrays.
    images = [to_numpy_array(image) for image in images]

    if is_scaled_image(images[0]) and do_rescale:
        logger.warning_once(
            "It looks like you are trying to rescale already rescaled images. If the input"
            " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        )

    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(images[0])

    if do_resize:
        images = [
            self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
            for image in images
        ]

    if do_center_crop:
        images = [
            self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
        ]

    if do_rescale:
        images = [
            self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
            for image in images
        ]

    if do_normalize:
        images = [
            self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
            for image in images
        ]

    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
    ]

    data = {"pixel_values": images}
    return BatchFeature(data=data, tensor_type=return_tensors)

class MyProcessor(CLIPProcessor):
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        tokenizer_kwargs, image_processor_kwargs = {}, {}
        if kwargs:
            tokenizer_kwargs = {k: v for k, v in kwargs.items() if k not in self.image_processor._valid_processor_keys}
            image_processor_kwargs = {
                k: v for k, v in kwargs.items() if k in self.image_processor._valid_processor_keys
            }

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **tokenizer_kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **image_processor_kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)