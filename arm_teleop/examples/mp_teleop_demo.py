from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation, EndEffectorTarget, GripperTarget
from observers.camera_viewer import CameraViewer

from sqlite_reader import SQLiteReader

DEBUG = True

class SqliteInput(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # self.DeclareAbstractOutputPort(
        #     "hand_positions", # name
        #     lambda: AbstractValue.Make(np.zeros([6])), # type
        #     self.calc, # calc
        # )
        self.DeclareVectorOutputPort(
            "hand_positions", # name
            # lambda: AbstractValue.Make(np.zeros([6])), # type; ORIG
            # np.array([np.pi,0.0,0.0,0.6,0.0,0.2]), # type; ALT1
            6, # type; ALT2
            self.calc, # calc
        )
        # Create a data reader instance
        self.data_reader = SQLiteReader()
        self.scale_factor = 5

    def calc(self, context, output):
        rows = self.data_reader.fetch_cube_positions(type="right_hand")
        data = np.zeros([6])
        # note: spaghetti code...
        for i, row in enumerate(rows):
            cube_id, x, y, z = row
            if cube_id == 16:
                x, y, z = (x*self.scale_factor, y*self.scale_factor, z*self.scale_factor)
                data = np.array([np.pi, 0.0, 0.0, x, y, z])
        output.set_value(data)

class SqlitePosInput(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort(
            "hand_positions", # name
            3, # size
            self.calc, # calc
        )
        # Create a data reader instance
        self.data_reader = SQLiteReader()
        self.scale_factor = 5

    def calc(self, context, output):
        rows = self.data_reader.fetch_cube_positions(type="right_hand")
        data = np.zeros([3])
        # note: spaghetti code...
        for i, row in enumerate(rows):
            cube_id, x, y, z = row
            if cube_id == 16:
                x, y, z = (x*self.scale_factor, y*self.scale_factor, z*self.scale_factor)
                data = np.array([x, y, z])
        output.set_value(data)


########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = False
# show_station_diagram = True

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
# show_toplevel_diagram = False
show_toplevel_diagram = True

# Run a quick simulation
simulate = True

# If we're running a simulation, choose which sort of commands are
# sent to the arm and the gripper
# ee_command_type = EndEffectorTarget.kTwist      # kPose, kTwist, or kWrench
ee_command_type = EndEffectorTarget.kPose      # kPose, kTwist, or kWrench
gripper_command_type = GripperTarget.kPosition  # kPosition or kVelocity

# If we're running a simulation, whether to include a simulated camera
# and show the associated image
include_camera = True
show_camera_window = False
# show_camera_window = True

# Which gripper to use (hande or 2f_85)
gripper_type = "hande"

########################################################################

# Set up the kinova station
station = KinovaStation(time_step=0.002, n_dof=7)
station.SetupSinglePegScenario(gripper_type=gripper_type, arm_damping=False)
if include_camera:
    station.AddCamera(show_window=show_camera_window)
    station.ConnectToMeshcatVisualizer()

if show_station_diagram:
    # Show the station's system diagram
    plt.figure()
    plot_system_graphviz(station,max_depth=1)
    plt.show()

# Connect input ports to the kinova station
builder = DiagramBuilder()
builder.AddSystem(station)

if DEBUG:
    # ORIG
    # # visualize target coordinate as a red sphere
    # target_sphere = Sphere(radius=0.01, color=np.array([1.0,0.0,0.0]))
    # target_sphere.set_name("target_sphere")
    # builder.Connect(
    #     builder.AddSystem(SqliteInput(pose_des)) # ALT1

    # )

    # CHATGPT
    from sphere import SphereSystem

    sphere_model_instance = station.plant.AddModelInstance("sphere")

    sphere_system = builder.AddSystem(SphereSystem(radius=0.05, initial_pose=np.array([0.0, 0.0, 0.1]), color=np.array([1.0, 0.0, 0.0, 1.0])))
    sphere_system.add_sphere_to_scene(station.plant, station.scene_graph, sphere_model_instance)

    # target_source = builder.AddSystem(SqlitePosInput())
    target_source = builder.AddSystem(SqliteInput())
    # builder.Connect(
    #     target_source.get_output_port(),
    #     # sphere_system.get_input_port()
    #     station.GetInputPort("viz_position")
    # )

builder.Connect( # ALT1
    target_source.get_output_port(),
    sphere_system.get_input_port()
    # station.GetInputPort("viz_position")
)

# # NOT SURE:
# builder.Connect(
#     station.scene_graph.get_query_output_port(),
#     station.plant.get_geometry_query_input_port())

# builder.ExportInput(station.scene_graph.GetInputPort("sphere_system_pose"))
# builder.ExportInput(builder.GetSubsystemByName("kinova_manipulation_station").GetInputPort("sphere_system_pose"))
# builder.ExportInput(builder.GetSubsystemByName("scene_graph").GetInputPort("sphere_system_pose"))
builder.ExportInput(builder.GetSubsystemByName("kinova_manipulation_station").GetSubsystemByName("scene_graph").GetInputPort("sphere_system_pose"))

builder.Connect(
    sphere_system.get_output_port(),
    # station.scene_graph.get_source_pose_port(station.plant.get_source_id()))
    station.scene_graph.GetInputPort("sphere_system_pose"))

station.Finalize()

# Set (constant) command to send to the system
if ee_command_type == EndEffectorTarget.kPose:
    pose_des = np.array([np.pi,0.0,0.0,
                            0.6,0.0,0.2])
    # target_source = builder.AddSystem(ConstantVectorSource(pose_des)) # ORIG
    target_source = builder.AddSystem(SqliteInput()) # ALT1

elif ee_command_type == EndEffectorTarget.kTwist:
    twist_des = np.array([0.0,0.1,0.0,
                          0.0,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(twist_des))

elif ee_command_type == EndEffectorTarget.kWrench:
    wrench_des = np.array([0.0,0.0,0.0,
                            0.0,0.0,0.0])
    target_source = builder.AddSystem(ConstantVectorSource(wrench_des))

else:
    raise RuntimeError("invalid end-effector target type")

# Send end-effector command and type
target_type_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(ee_command_type)))
builder.Connect(
        target_type_source.get_output_port(),
        station.GetInputPort("ee_target_type"))

builder.Connect(
        target_source.get_output_port(),
        station.GetInputPort("ee_target"))

target_source.set_name("ee_command_source")
target_type_source.set_name("ee_type_source")

# Set gripper command
if gripper_command_type == GripperTarget.kPosition:
    q_grip_des = np.array([0])   # open at 0, closed at 1
    gripper_target_source = builder.AddSystem(ConstantVectorSource(q_grip_des))

elif gripper_command_type == GripperTarget.kVelocity:
    v_grip_des = np.array([1.0])
    gripper_target_source = builder.AddSystem(ConstantVectorSource(v_grip_des))

# Send gripper command and type
gripper_target_type_source = builder.AddSystem(ConstantValueSource(
                                         AbstractValue.Make(gripper_command_type)))
builder.Connect(
        gripper_target_type_source.get_output_port(),
        station.GetInputPort("gripper_target_type"))

builder.Connect(
        gripper_target_source.get_output_port(),
        station.GetInputPort("gripper_target"))

gripper_target_source.set_name("gripper_command_source")
gripper_target_type_source.set_name("gripper_type_source")

# Loggers force certain outputs to be computed
wrench_logger = LogVectorOutput(station.GetOutputPort("measured_ee_wrench"),builder)
wrench_logger.set_name("wrench_logger")

pose_logger = LogVectorOutput(station.GetOutputPort("measured_ee_pose"), builder)
pose_logger.set_name("pose_logger")

twist_logger = LogVectorOutput(station.GetOutputPort("measured_ee_twist"), builder)
twist_logger.set_name("twist_logger")

gripper_logger = LogVectorOutput(station.GetOutputPort("measured_gripper_velocity"), builder)
gripper_logger.set_name("gripper_logger")

if include_camera:
    # Camera observer allows us to access camera data, and must be connected
    # to view the camera stream.
    camera_viewer = builder.AddSystem(CameraViewer())
    camera_viewer.set_name("camera_viewer")

    builder.Connect(
            station.GetOutputPort("camera_rgb_image"),
            camera_viewer.GetInputPort("color_image"))
    builder.Connect(
            station.GetOutputPort("camera_depth_image"),
            camera_viewer.GetInputPort("depth_image"))

    # Convert the depth image to a point cloud
    point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                        CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                        fields=BaseField.kXYZs | BaseField.kRGBs))
    point_cloud_generator.set_name("point_cloud_generator")
    builder.Connect(
            station.GetOutputPort("camera_depth_image"),
            point_cloud_generator.depth_image_input_port())

    # Connect camera pose to point cloud generator
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            point_cloud_generator.GetInputPort("camera_pose"))

    # Visualize the point cloud with meshcat
    meshcat_point_cloud = builder.AddSystem(
            MeshcatPointCloudVisualizer(station.meshcat, "point_cloud", 0.2))
    meshcat_point_cloud.set_name("point_cloud_viz")
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            meshcat_point_cloud.cloud_input_port())

# Build the system diagram
diagram = builder.Build()
diagram.set_name("toplevel_system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    # Show the overall system diagram
    plt.figure()
    # plot_system_graphviz(diagram,max_depth=1)
    plot_system_graphviz(diagram,max_depth=3)
    plt.savefig("./top_level_diagram_viz.jpg", dpi=1000)
    plt.show()

if simulate:
    # Set default arm positions
    station.go_home(diagram, diagram_context, name="Home")

    # Set starting position for any objects in the scene
    station.SetManipulandStartPositions(diagram, diagram_context)

    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    # simulator.set_target_realtime_rate(1.0)
    simulator.set_target_realtime_rate(5.0)
    simulator.set_publish_every_time_step(False)

    # Run simulation
    simulator.Initialize()
    # simulator.AdvanceTo(5.0)
    simulator.AdvanceTo(60.0)
