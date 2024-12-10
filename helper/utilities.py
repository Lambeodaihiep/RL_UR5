import pybullet as p
import glob
import logging
import numpy as np

# Cionfig logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - func:%(funcName)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Models:
    """
    Abstract class representing a model that manages objects or data.

    This class serves as a blueprint for specific model implementations.
    It defines common methods such as loading objects, retrieving the length of
    the model, and accessing specific items, but does not implement their
    functionality directly. Subclasses must implement these methods.
    """

    def load_objects(self):
        """
        Abstract method for loading objects into the model.

        This method should be implemented by subclasses to define how
        objects or data are loaded into the model. The implementation
        will depend on the specific type of model being used.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Abstract method for retrieving the length of the model.

        This method should return the number of objects or items in the model.
        The exact behavior depends on the model's implementation.

        Returns:
            int: The length of the model (number of objects/items).

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """
        Abstract method for retrieving an item from the model.

        This method should be implemented by subclasses to specify how
        to retrieve an item from the model, typically by index or key.

        Parameters:
            item: The index or key of the item to retrieve.

        Returns:
            The item corresponding to the given index or key.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        return NotImplementedError


class YCBModels(Models):
    """
    Class representing the YCB model objects, inheriting from the Models base class.

    This class is used to load YCB object models (using mesh data), handle their visualization,
    and collision shapes. It supports filtering the loaded models based on selected names.

    Attributes:
        obj_files (list): List of object file paths (meshes).
        selected_names (tuple): Tuple of object names to filter specific objects.
        visual_shapes (list): List of visual shapes corresponding to loaded models.
        collision_shapes (list): List of collision shapes corresponding to loaded models.
    """

    def __init__(self, root, selected_names: tuple = ()):
        """
        Initialize the YCBModels class.

        Parameters:
            root (str): Path pattern to search for object files (e.g., "*.obj").
            selected_names (tuple, optional): Tuple of names of specific objects to load.
                If empty, all objects matching the root pattern are loaded.
        """
        self.obj_files = glob.glob(root)
        self.selected_names = selected_names

        self.visual_shapes = []
        self.collision_shapes = []

    def load_objects(self):
        """
        Load the objects from files and create collision and visual shapes.

        This method loops through the object files, filters them based on the selected names,
        and creates both collision and visual shapes for each object using PyBullet.

        Parameters:
            None

        Returns:
            None
        """

        shift = [0, 0, 0]  # Default shift for positioning the shapes
        mesh_scale = [1, 1, 1]  # Default scale for the meshes

        for filename in self.obj_files:
            # Check selected_names
            if self.selected_names:
                in_selected = False
                for name in self.selected_names:
                    if name in filename:
                        in_selected = True
                if not in_selected:
                    continue
            logger.info("Loading %s" % filename)
            # Create collision shape
            self.collision_shapes.append(
                p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=filename,
                    collisionFramePosition=shift,
                    meshScale=mesh_scale,
                )
            )
            # Create visual shape
            self.visual_shapes.append(
                p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=filename,
                    visualFramePosition=shift,
                    meshScale=mesh_scale,
                )
            )

    def __len__(self):
        """
        Return the number of objects loaded (based on collision shapes).

        This method returns the length of the loaded objects, which is equivalent
        to the number of collision shapes (as each object has both a visual and a collision shape).

        Parameters:
            None

        Returns:
            int: The number of objects (based on collision shapes).
        """
        return len(self.collision_shapes)

    def __getitem__(self, idx):
        """
        Get the visual and collision shapes for a given object by index.

        This method allows indexing into the YCB model objects to retrieve both the
        visual and collision shapes for a specific object.

        Parameters:
            idx (int): The index of the object in the shapes list.

        Returns:
            tuple: A tuple containing the visual shape and collision shape for the object.
        """
        return self.visual_shapes[idx], self.collision_shapes[idx]


class Camera:
    """
    A class that represents a 3D camera in a simulation environment.

    The Camera class is used to simulate a camera in a 3D environment and provides methods
    to capture RGB, depth, and segmentation images. It also provides functionality to convert
    pixel coordinates to world coordinates.

    Attributes:
        width (int): Width of the camera image.
        height (int): Height of the camera image.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        fov (float): Field of view of the camera in degrees.
        view_matrix (list): View matrix for the camera.
        projection_matrix (list): Projection matrix for the camera.
        tran_pix_world (ndarray): Transformation matrix to convert from pixel coordinates to world coordinates.
    """

    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        """
        Initializes the Camera object with specified parameters.

        Args:
            cam_pos (list): Position of the camera in the 3D space.
            cam_tar (list): Target (looking) point of the camera in the 3D space.
            cam_up_vector (list): Up vector of the camera, typically [0, 1, 0].
            near (float): Near clipping plane distance.
            far (float): Far clipping plane distance.
            size (tuple): Size of the camera image as (width, height).
            fov (float): Field of view of the camera in degrees.

        Sets up the camera's view matrix, projection matrix, and transformation matrix.
        """
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, aspect, self.near, self.far
        )

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order="F")
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order="F")
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        """
        Converts pixel coordinates and depth to world coordinates.

        Args:
            w (int): The x-coordinate (pixel) in the image.
            h (int): The y-coordinate (pixel) in the image.
            d (float): The depth value (between 0 and 1).

        Returns:
            ndarray: A 3D position in the world space corresponding to the given pixel coordinates
                      and depth value.
        """
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        """
        Captures an RGB image, depth image, and segmentation image from the camera.

        Returns:
            tuple: A tuple containing three elements:
                - rgb (ndarray): The RGB image captured by the camera.
                - depth (ndarray): The depth image captured by the camera.
                - seg (ndarray): The segmentation image captured by the camera.
        """
        _w, _h, rgb, depth, seg = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
        )
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        """
        Converts a depth map to world coordinates for all pixels in the image.
        reference: https://stackoverflow.com/a/62247245

        Args:
            depth (ndarray): A 2D array representing the depth map of the image.

        Returns:
            ndarray: A 3D array where each element contains the world coordinates of the corresponding
                      pixel in the depth map.
        """
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array(
            [x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]
        ).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
