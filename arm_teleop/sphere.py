import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    RigidTransform,
    RollPitchYaw,
    Sphere,
    SceneGraph,
    Simulator,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    SpatialInertia,
    UnitInertia,
    CoulombFriction,
    Rgba,
    LeafSystem,
    BasicVector,
    AbstractValue,
    PortDataType,
    GeometryInstance,
)

class SphereSystem(LeafSystem):
    # def __init__(self, radius=0.1, initial_pose=np.array([0.0, 0.0, 0.0]), color=Rgba(1, 0, 0, 1)):
    def __init__(self, radius=0.05, initial_pose=np.array([0.0, 0.0, 0.0]), color=np.array([1.0, 0.0, 0.0, 1.0])):
        super().__init__()
        self.radius = radius
        self.color = color

        # Declare an input port for the sphere's position
        # self.DeclareVectorInputPort("viz_position", BasicVector(3)) # ORIG
        self.DeclareVectorInputPort("viz_position", BasicVector(6)) # ALT1

        # Declare a pose output port for the scene graph
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcPoseOutput
        )

        # Store the initial pose
        self.initial_pose = RigidTransform(p=initial_pose)

    def CalcPoseOutput(self, context, output):
        # Get the current input position
        # position = self.get_input_port(0).Eval(context) # ORIG
        position = self.get_input_port(0).Eval(context)[:3] # ALT1

        # Create a new pose based on the input position
        new_pose = RigidTransform(p=position)

        # Set the output to the new pose
        output.set_value(new_pose)

    # def add_sphere_to_scene(self, plant, scene_graph):
    def add_sphere_to_scene(self, plant, scene_graph, sphere_model_instance):
        # Define the inertia properties of the sphere (assuming uniform density)
        mass = 1.0  # arbitrary mass
        inertia = UnitInertia.SolidSphere(self.radius)
        spatial_inertia = SpatialInertia(mass, np.zeros(3), inertia)

        # Add the sphere to the multibody plant
        # body = plant.AddRigidBody("sphere", spatial_inertia)
        body = plant.AddRigidBody("sphere", sphere_model_instance, spatial_inertia)
        shape = Sphere(self.radius)
        geometry_instance = GeometryInstance(RigidTransform(), shape, "sphere")

        plant.RegisterCollisionGeometry(body, RigidTransform(), shape, "collision", CoulombFriction(0.9, 0.8))
        plant.RegisterVisualGeometry(body, RigidTransform(), shape, "visual", self.color)

        # Connect the sphere's position to the world frame, updated via the pose output port
        plant.WeldFrames(
            plant.world_frame(),
            body.body_frame(),
            self.initial_pose
        )

        # Register the output port of the system to update the pose
        source_id = scene_graph.RegisterSource("sphere_system")
        # scene_graph.RegisterAnchoredGeometry("sphere_system", shape, new_pose=RigidTransform())
        # scene_graph.RegisterAnchoredGeometry("sphere_system", shape)
        # scene_graph.RegisterAnchoredGeometry("sphere_system", geometry_instance)
        scene_graph.RegisterAnchoredGeometry(source_id, geometry_instance)
