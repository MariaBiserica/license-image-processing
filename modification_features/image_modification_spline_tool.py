import numpy as np
from PIL import Image


class PointwiseOperations:
    @staticmethod
    def calculate_hermite_curve(control_points):
        hermite_curve = []

        if len(control_points) < 2:
            raise ValueError("At least two control points are required for Hermite interpolation.")

        for i in range(len(control_points) - 1):
            p0 = control_points[i]
            p1 = control_points[i + 1]

            if i == 0:
                p_minus1 = (-p0[0], p0[1])
            else:
                p_minus1 = control_points[i - 1]

            if i == len(control_points) - 2:
                p2 = (2 * p1[0] - p0[0], 2 * p1[1] - p0[1])
            else:
                p2 = control_points[i + 2]

            m0 = ((p1[0] - p_minus1[0]) / 2, (p1[1] - p_minus1[1]) / 2)
            m1 = ((p2[0] - p0[0]) / 2, (p2[1] - p0[1]) / 2)

            for t in np.arange(0, 1, 0.01):
                h1 = 2 * t ** 3 - 3 * t ** 2 + 1
                h2 = -2 * t ** 3 + 3 * t ** 2
                h3 = t ** 3 - 2 * t ** 2 + t
                h4 = t ** 3 - t ** 2

                x = int(h1 * p0[0] + h2 * p1[0] + h3 * m0[0] + h4 * m1[0])
                y = int(h1 * p0[1] + h2 * p1[1] + h3 * m0[1] + h4 * m1[1])

                hermite_curve.append((x, y))

        hermite_curve.append(control_points[-1])
        return hermite_curve

    @staticmethod
    def apply_lut(original_values, hermite_curve):
        lut = PointwiseOperations.build_lut(hermite_curve)
        result_values = [lut[min(max(0, val), 255)] for val in original_values]
        return result_values

    @staticmethod
    def build_lut(hermite_curve):
        lut = [0] * 256
        for point in hermite_curve:
            lut[min(max(0, point[0]), 255)] = min(max(0, point[1]), 255)
        return lut


def modify_image(image, control_points):
    pixels = np.array(image)

    flat_pixels = pixels.flatten()
    hermite_curve = PointwiseOperations.calculate_hermite_curve(control_points)
    modified_pixels = PointwiseOperations.apply_lut(flat_pixels, hermite_curve)
    modified_image = np.reshape(modified_pixels, pixels.shape)

    return Image.fromarray(modified_image.astype(np.uint8))
