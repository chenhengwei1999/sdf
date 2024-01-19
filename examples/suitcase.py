from sdf import *

def suitcase():
    # Create the main body of the suitcase
    body = rounded_box([15, 10, 3], 0.5)

    # Add the handle
    handle = cylinder(1.5).translate((0, 5, 1.5)) & slab(z0=0, z1=12)

    # Subtract the space for wheels
    body -= slab(z0=-1, z1=0).translate((2, -5, -0.5)).circular_array(4, 2.5)

    # Combine the body and handle using union
    suitcase_model = union(body, handle)

    return suitcase_model

if __name__ == "__main__":
    suitcase_model = suitcase()
    suitcase_model.save('suitcase.obj', samples=2**26)
