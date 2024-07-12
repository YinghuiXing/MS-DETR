# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 15:53
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : visualizer.py
from typing import Optional, Union, List, Any, Type, Tuple
from .palette import get_palette, _get_adaptive_scales
import numpy as np
import torch
import os
from .utils import img_from_canvas, check_type, check_type_and_length, tensor2ndarray, value2list, imwrite, imread


classes = ('person', 'car', 'van', 'bus', 'truck', 'cyclist', 'e-cyclist', 'freight-car')
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)]
# 蓝色，，


def color_val_matplotlib(
    colors: Union[str, tuple, List[Union[str, tuple]]]
) -> Union[str, tuple, List[Union[str, tuple]]]:
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
    Returns:
        Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
        floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [
            color_val_matplotlib(color)  # type:ignore
            for color in colors
        ]
        return colors
    else:
        raise TypeError(f'Invalid type for color: {type(colors)}')


class Visualizer:
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 fig_save_cfg=dict(frameon=False),
                 fig_show_cfg=dict(frameon=False)):
        self.fig_save = None
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg

        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()

        if image is not None:
            self.set_image(image)

    def _initialize_fig(self, fig_cfg) -> tuple:
        """Build figure according to fig_cfg.

        Args:
            fig_cfg (dict): The config to build figure.

        Returns:
             tuple: build canvas figure and axes.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)  # 隐藏坐标轴

        # remove white edges by set subplot margin
        # 调整Figure对象的子图布局的方法
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(fig)
        return canvas, fig, ax

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')

        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]  # 在np.ndarray中, 图像的形式为H * W * 3
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        self.fig_save.set_size_inches(  # type: ignore
            (self.width + 1e-2) / self.dpi, (self.height + 1e-2) / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()  # 用于清空（clear）当前Axes对象的内容。
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')

    def get_image(self) -> np.ndarray:
        """Get the drawn image. The format is RGB.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        return img_from_canvas(self.fig_save_canvas)   # type: ignore

    def draw_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw with
                the format of(x1,y1,x2,y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 0.8.
        """
        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))
        bboxes = tensor2ndarray(bboxes)

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()

        poly = np.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            axis=-1).reshape(-1, 4, 2)
        poly = [p for p in poly]
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

    def draw_polygons(
        self,
        polygons: Union[Union[np.ndarray, torch.Tensor],
                        List[Union[np.ndarray, torch.Tensor]]],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],\
                List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                with the format of (x1,y1,x2,y2,...,xn,yn).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of polygons. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of polygons.
                Defaults to 0.8.
        """
        from matplotlib.collections import PolyCollection
        check_type('polygons', polygons, (list, np.ndarray, torch.Tensor))
        edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
        face_colors = color_val_matplotlib(face_colors)  # type: ignore

        if isinstance(polygons, (np.ndarray, torch.Tensor)):
            polygons = [polygons]
        if isinstance(polygons, list):
            for polygon in polygons:
                assert polygon.shape[1] == 2, (
                    'The shape of each polygon in `polygons` should be (M, 2),'
                    f' but got {polygon.shape}')
        polygons = [tensor2ndarray(polygon) for polygon in polygons]
        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(polygons)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        polygon_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths)

        self.ax_save.add_collection(polygon_collection)
        return self

    def draw_texts(
        self,
        texts: Union[str, List[str]],
        positions: Union[np.ndarray, torch.Tensor],
        font_sizes: Optional[Union[int, List[int]]] = None,
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        vertical_alignments: Union[str, List[str]] = 'top',
        horizontal_alignments: Union[str, List[str]] = 'left',
        font_families: Union[str, List[str]] = 'sans-serif',
        bboxes: Optional[Union[dict, List[dict]]] = None,
        font_properties: Optional[Union['FontProperties',
                                        List['FontProperties']]] = None
    ) -> 'Visualizer':
        """Draw single or multiple text boxes.

        Args:
            texts (Union[str, List[str]]): Texts to draw.
            positions (Union[np.ndarray, torch.Tensor]): The position to draw
                the texts, which should have the same length with texts and
                each dim contain x and y.
            font_sizes (Union[int, List[int]], optional): The font size of
                texts. ``font_sizes`` can have the same length with texts or
                just single value. If ``font_sizes`` is single value, all the
                texts will have the same font size. Defaults to None.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors
                of texts. ``colors`` can have the same length with texts or
                just single value. If ``colors`` is single value, all the
                texts will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            vertical_alignments (Union[str, List[str]]): The verticalalignment
                of texts. verticalalignment controls whether the y positional
                argument for the text indicates the bottom, center or top side
                of the text bounding box.
                ``vertical_alignments`` can have the same length with
                texts or just single value. If ``vertical_alignments`` is
                single value, all the texts will have the same
                verticalalignment. verticalalignment can be 'center' or
                'top', 'bottom' or 'baseline'. Defaults to 'top'.
            horizontal_alignments (Union[str, List[str]]): The
                horizontalalignment of texts. Horizontalalignment controls
                whether the x positional argument for the text indicates the
                left, center or right side of the text bounding box.
                ``horizontal_alignments`` can have
                the same length with texts or just single value.
                If ``horizontal_alignments`` is single value, all the texts
                will have the same horizontalalignment. Horizontalalignment
                can be 'center','right' or 'left'. Defaults to 'left'.
            font_families (Union[str, List[str]]): The font family of
                texts. ``font_families`` can have the same length with texts or
                just single value. If ``font_families`` is single value, all
                the texts will have the same font family.
                font_familiy can be 'serif', 'sans-serif', 'cursive', 'fantasy'
                or 'monospace'.  Defaults to 'sans-serif'.
            bboxes (Union[dict, List[dict]], optional): The bounding box of the
                texts. If bboxes is None, there are no bounding box around
                texts. ``bboxes`` can have the same length with texts or
                just single value. If ``bboxes`` is single value, all
                the texts will have the same bbox. Reference to
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
                for more details. Defaults to None.
            font_properties (Union[FontProperties, List[FontProperties]], optional):
                The font properties of texts. FontProperties is
                a ``font_manager.FontProperties()`` object.
                If you want to draw Chinese texts, you need to prepare
                a font file that can show Chinese characters properly.
                For example: `simhei.ttf`, `simsun.ttc`, `simkai.ttf` and so on.
                Then set ``font_properties=matplotlib.font_manager.FontProperties(fname='path/to/font_file')``
                ``font_properties`` can have the same length with texts or
                just single value. If ``font_properties`` is single value,
                all the texts will have the same font properties.
                Defaults to None.
                `New in version 0.6.0.`
        """  # noqa: E501
        from matplotlib.font_manager import FontProperties
        check_type('texts', texts, (str, list))
        if isinstance(texts, str):
            texts = [texts]
        num_text = len(texts)
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)
        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape == (num_text, 2), (
            '`positions` should have the shape of '
            f'({num_text}, 2), but got {positions.shape}')
        positions = positions.tolist()

        if font_sizes is None:
            font_sizes = self._default_font_size
        check_type_and_length('font_sizes', font_sizes, (int, float, list),
                              num_text)
        font_sizes = value2list(font_sizes, (int, float), num_text)

        check_type_and_length('colors', colors, (str, tuple, list), num_text)
        colors = value2list(colors, (str, tuple), num_text)
        colors = color_val_matplotlib(colors)  # type: ignore

        check_type_and_length('vertical_alignments', vertical_alignments,
                              (str, list), num_text)
        vertical_alignments = value2list(vertical_alignments, str, num_text)

        check_type_and_length('horizontal_alignments', horizontal_alignments,
                              (str, list), num_text)
        horizontal_alignments = value2list(horizontal_alignments, str,
                                           num_text)

        check_type_and_length('font_families', font_families, (str, list),
                              num_text)
        font_families = value2list(font_families, str, num_text)

        if font_properties is None:
            font_properties = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('font_properties', font_properties,
                                  (FontProperties, list), num_text)
            font_properties = value2list(font_properties, FontProperties,
                                         num_text)

        if bboxes is None:
            bboxes = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('bboxes', bboxes, (dict, list), num_text)
            bboxes = value2list(bboxes, dict, num_text)

        for i in range(num_text):
            self.ax_save.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i],  # type: ignore
                bbox=bboxes[i],  # type: ignore
                verticalalignment=vertical_alignments[i],
                horizontalalignment=horizontal_alignments[i],
                family=font_families[i],
                fontproperties=font_properties[i],
                color=colors[i])
        return self


def draw_bboxes_4_single_image(image: np.ndarray,
                                    bboxes: np.ndarray,
                                    bboxes_color: Optional[Union[str, Tuple[int]]],
                                    info: Optional[Union[str, List[str]]],
                                    visualizer: Visualizer,
                                    text_color: Optional[Union[str, Tuple[int]]] = (200, 200, 200),
                                    alpha: float = 0.8,
                                    line_width: Union[int, float] = 3,
                                    save: bool = False,
                                    save_dir: str = None) -> np.ndarray:
    bboxes_num = bboxes.shape[0]
    visualizer.set_image(image)

    bbox_palette = get_palette(palette, len(classes))
    if bboxes_color is None:
        bboxes_color = [bbox_palette[0] for _ in range(bboxes_num)]

    #print(bboxes_color)

    # line_width = [5 if color[0] == 0 and color[1] == 0 and color[2] == 255 else 3 for color in bboxes_color]
    #print(line_width)
    print(bboxes.shape[0] == len(bboxes_color))

    visualizer.draw_bboxes(bboxes, edge_colors=bboxes_color, alpha=alpha, line_widths=line_width)

    if info is not None:
        text_palette = get_palette(text_color, len(classes))
        text_colors = [text_palette[0] for _ in range(bboxes_num)]

        positions = bboxes[:, :2] + line_width
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)

        if isinstance(info, str):
            info = [str] * bboxes_num

        for i, (pos, text) in enumerate(zip(positions, info)):
            visualizer.draw_texts(
                text,
                pos,
                colors=text_colors[i],
                font_sizes=int(13 * scales[i]),
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }],
                font_families = 'serif'
            )

    return visualizer.get_image()


def vis_annotation_4_two_image(name: str,
                       image_1: np.ndarray,
                       image_2: np.ndarray,
                       bboxes_1: np.ndarray,
                       bboxes_color_1,
                       info_1: Optional[Union[str, List[str]]],
                       bboxes_2: np.ndarray,
                       bboxes_color_2,
                       info_2: Optional[Union[str, List[str]]],
                       visualizer: Visualizer,
                       save_dir: str,
                       concat_axis: int = 0,
                       ) -> None:
    image_1 = image_1.clip(0, 255).astype(np.uint8)
    image_2 = image_2.clip(0, 255).astype(np.uint8)

    image_1 = draw_bboxes_4_single_image(image_1, bboxes_1, bboxes_color_1, info_1, visualizer=visualizer)
    image_2 = draw_bboxes_4_single_image(image_2, bboxes_2, bboxes_color_2, info_2, visualizer=visualizer)

    img = np.concatenate((image_1, image_2), axis=concat_axis)  # concat_axis==0表示垂直，==1表示水平

    img_path = os.path.join(save_dir, name)
    imwrite(img, img_path)

def vis_annotation_4_three_image(name: str,
                       image_1: np.ndarray,
                       image_2: np.ndarray,
                       image_3: np.ndarray,
                       visualizer: Visualizer,
                       save_dir: str,
                       concat_axis: int = 1,
                       ) -> None:
    image_1 = image_1.clip(0, 255).astype(np.uint8)
    image_2 = image_2.clip(0, 255).astype(np.uint8)
    image_3 = image_3.clip(0, 255).astype(np.uint8)

    image_1 = draw_bboxes_4_single_image(image_1, np.zeros((0, 4)), None, None, visualizer=visualizer)
    image_2 = draw_bboxes_4_single_image(image_2, np.zeros((0, 4)), None, None, visualizer=visualizer)
    image_3 = draw_bboxes_4_single_image(image_3, np.zeros((0, 4)), None, None, visualizer=visualizer)

    img = np.concatenate((image_1, image_2, image_3), axis=concat_axis)  # concat_axis==0表示垂直，==1表示水平

    img_path = os.path.join(save_dir, name)
    imwrite(img, img_path)

