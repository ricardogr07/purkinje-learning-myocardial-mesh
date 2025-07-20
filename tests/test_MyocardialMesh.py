from myocardial_mesh import MyocardialMesh

mesh = MyocardialMesh(
    myo_mesh='path/to/myo.vtk',
    fibers='path/to/fiber.vtk',
    electrodes_position='path/to/electrodes.pkl'
)
print(mesh.K.shape)