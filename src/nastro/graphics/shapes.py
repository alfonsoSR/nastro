from matplotlib import patches as mpatches
from ..types import Vector, is_vector, Double
import numpy as np


class Circle(mpatches.Circle):

    def __init__(self, x: float, y: float, radius: float) -> None:

        super().__init__((x, y), radius=radius)

        return None


class Sphere:

    def __init__(self, x: float, y: float, z: float, radius: float) -> None:

        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

        return None

    def surface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x = self.x + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.y + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.z + self.radius * np.outer(np.ones_like(u), np.cos(v))

        return x, y, z
