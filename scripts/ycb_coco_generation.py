import blenderproc as bproc
import argparse
import os
import numpy as np
import json 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bop_parent_path",
    default="resources",
    help="Path to the bop datasets parent directory",
)
parser.add_argument(
    "--cc_textures_path",
    default="resources/cc_textures",
    help="Path to downloaded cc textures",
)
parser.add_argument(
    "--output_dir",
    "-o",
    default="output",
    help="Path to where the final files will be saved ",
)
parser.add_argument(
    "--num_scenes",
    type=int,
    default=1,
    help="How many scenes with 25 images each to generate",
)
args = parser.parse_args()

bproc.init()

# load distractor bop objects
hope_targets = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_parent_path, "hope"), mm2m=True
)

# hope_targets = list(
#     np.random.choice(hope_targets, size=10, replace=False)
# )




# load bop objects into the scene
ycb_distractors = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_parent_path, "ycbv"), mm2m=True
)

# Hide all objects initially 
for obj in hope_targets + ycb_distractors:
    obj.hide(True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(
    bop_dataset_path=os.path.join(args.bop_parent_path, "ycbv")
)

hope_targets = hope_targets[:1]


all_objs = hope_targets + ycb_distractors

with open(os.path.join(args.bop_parent_path, "bop_objects_name_map.json"), "r") as map_file:

    name_mapping = json.load(map_file)

print(f"name_mapping: {name_mapping}")


# set shading and hide objects
# for obj in (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs):
i = 1 
for _, obj in enumerate(all_objs):
    obj.set_shading_mode("auto")
    if obj in hope_targets:
        obj.set_cp("category_id", i)
        i += 1 

        dataset = obj.get_cp("bop_dataset_name")
        name = obj.get_name().split('.')[0]
        print(f"name: {name}")

        if dataset in name_mapping and name in name_mapping[dataset]:
            new_name = name_mapping[dataset][name] + f"_{dataset}"
            print(f"renaming object {name} to {new_name}" )
            obj.set_name(new_name)
    else: 
        obj.set_cp("category_id", 0)
        obj.set_name("None")


# create room
room_planes = [
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
    ),
    bproc.object.create_primitive(
        "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
    ),
]
for plane in room_planes:
    plane.enable_rigidbody(
        False,
        collision_shape="BOX",
        mass=1.0,
        friction=100.0,
        linear_damping=0.99,
        angular_damping=0.99,
    )

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive(
    "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
)
light_plane.set_name("light_plane")
light_plane_material = bproc.material.create("light_material")

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


bproc.renderer.set_max_amount_of_samples(50)
# activate normal rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance", "name"], default_values={"category_id": 0}
)

for i in range(args.num_scenes):
    
    sampled_target_bop_objs = hope_targets
    # Sample bop objects for a scene
    sampled_distractor_bop_objs = list(
        np.random.choice(ycb_distractors, size=12, replace=False)
    )



    # Randomize materials and set physics
    for obj in sampled_target_bop_objs + sampled_distractor_bop_objs:
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ["itodd", "tless"]:
            grey_col = np.random.uniform(0.1, 0.9)
            mat.set_principled_shader_value(
                "Base Color", [grey_col, grey_col, grey_col, 1]
            )
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(
            True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99
        )
        obj.hide(False)

    # Sample two light sources
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=1,
        radius_max=1.5,
        elevation_min=5,
        elevation_max=89,
    )
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions
    bproc.object.sample_poses(
        objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
        sample_pose_func=sample_pose_func,
        max_tries=1000,
    )

    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=10,
        check_object_interval=1,
        substeps_per_frame=20,
        solver_iters=25,
    )
    # define the camera intrinsics
    bproc.camera.set_resolution(512, 512)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
        sampled_target_bop_objs + sampled_distractor_bop_objs
    )

    cam_poses = 0
    while cam_poses < 20:
        # Sample location
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=0.61,
            radius_max=1.24,
            elevation_min=5,
            elevation_max=89,
        )
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(
            np.random.choice(sampled_target_bop_objs, size=len(hope_targets), replace=False)
        )
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159)
        )
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(
            cam2world_matrix, {"min": 0.3}, bop_bvh_tree
        ):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_coco_annotations(
        os.path.join(args.output_dir, "coco_data"),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    for obj in sampled_target_bop_objs + sampled_distractor_bop_objs:
        obj.disable_rigidbody()
        obj.hide(True)
