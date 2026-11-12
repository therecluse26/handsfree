import numpy as np
import pytest

from handsfree import eyenorm


def project(p, focal, size):
    return np.array([size[0] / 2 + focal * p[0] / -p[2], size[1] / 2 - focal * p[1] / -p[2]])


def warp_point(view, px):
    q = view.warp @ [px[0], px[1], 1.0]
    return q[:2] / q[2]


CENTRE = np.array(eyenorm.SIZE) / 2


@pytest.mark.parametrize("focal,size", [(1313.0, (1920, 1080)), (700.0, (1280, 720))])
@pytest.mark.parametrize("eye", [(-30.0, -100.0, -550.0), (250.0, 60.0, -400.0), (-300.0, -200.0, -850.0)])
def test_the_eye_is_at_the_centre_with_a_fixed_scale(focal, size, eye):
    eye = np.array(eye)
    other = eye + [63.0, 4.0, 0.0]  # a little roll
    view = eyenorm.eye_view(eye, other, focal, size, is_image_left=True)
    assert warp_point(view, project(eye, focal, size)) == pytest.approx(CENTRE, abs=1e-6)
    # A point 10 mm towards the other eye, at the same distance from the camera, is 20 px to the right.
    along = (other - eye) / np.linalg.norm(other - eye)
    across = along - (along @ eye) * eye / (eye @ eye)
    across /= np.linalg.norm(across)
    p = warp_point(view, project(eye + 10.0 * across, focal, size))
    assert p == pytest.approx(CENTRE + [10.0 * eyenorm.PX_PER_MM, 0.0], abs=0.15)


def test_the_image_right_eye_also_has_the_other_eye_towards_the_centre_of_the_face():
    eye, other = np.array([40.0, -100.0, -500.0]), np.array([-23.0, -100.0, -500.0])
    view = eyenorm.eye_view(eye, other, 1313.0, (1920, 1080), is_image_left=False)
    p = warp_point(view, project(other, 1313.0, (1920, 1080)))
    assert p[0] < CENTRE[0] - 100 and abs(p[1] - CENTRE[1]) < 1  # the other eye is to the left


def test_angles():
    eye, other = np.array([200.0, -150.0, -600.0]), np.array([263.0, -150.0, -600.0])
    view = eyenorm.eye_view(eye, other, 1313.0, (1920, 1080), is_image_left=True)
    assert view.angles(-eye) == pytest.approx([0.0, 0.0], abs=1e-9)  # a look into the camera
    up = view.angles(np.array([0.0, 100.0, 0.0]) - eye)
    right = view.angles(np.array([100.0, 0.0, 0.0]) - eye)
    assert up[1] > 5 and abs(up[0]) < 1 and right[0] > 5
    for yp in ([12.0, -7.0], [-25.0, 15.0]):
        assert view.angles(view.direction(np.array(yp))) == pytest.approx(yp, abs=1e-9)
    assert np.linalg.det(view.rotation) == pytest.approx(1.0)


def test_image_and_model_input():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    eye = np.array([-30.0, -100.0, -550.0])
    u, v = project(eye, 1313.0, (1920, 1080))
    cv2.circle(frame, (int(round(u)), int(round(v))), 6, (255, 255, 255), -1)
    view = eyenorm.eye_view(eye, eye + [63.0, 0.0, 0.0], 1313.0, (1920, 1080), is_image_left=True)
    whole = view.image(frame)
    assert whole.shape == (64, 96) and whole[32, 48] > 200 and whole[5, 5] == 0
    x0, y0 = int(u) - 96, int(v) - 64
    crop = view.image(frame[y0 : y0 + 128, x0 : x0 + 192], origin=(x0, y0))
    assert np.abs(crop.astype(int) - whole).mean() < 1.0  # a crop gives the same image as the full frame
    x = eyenorm.to_model_input(np.stack([whole, whole]), flip=[False, True])
    assert x.shape == (2, 1, 64, 96) and abs(float(x[0].mean())) < 1e-4
    assert np.allclose(x[1, 0], x[0, 0, :, ::-1])
