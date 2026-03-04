import numpy as np
import os

def run_nms_test(name, boxes, scores, iou_threshold, expected_indices, nms_function = None):
    """Run a single NMS test case. Returns True if passed, False otherwise."""
    print(f"\n{name}")
    print("-" * len(name))
    try:
        if nms_function is None:
            nms_function = lambda boxes, scores, iou_threshold: []

        kept = nms_function(boxes, scores, iou_threshold=iou_threshold)
        print("Returned:", kept)
        print("Expected:", expected_indices)

        is_valid_type = isinstance(kept, list) and all(isinstance(i, (int, np.integer)) for i in kept)
        is_correct = is_valid_type and kept == expected_indices

        print("PASS" if is_correct else "FAIL")
        return is_correct
    except Exception as e:
        print("ERROR:", repr(e))
        print("FAIL")
        return False

def run_nms_tests(nms_function = lambda boxes, scores, iou_threshold: []):
    results = []

    boxes = np.empty((0, 4), dtype=float)
    scores = np.empty((0,), dtype=float)
    results.append(run_nms_test(
        name="Test 1: empty input",
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        expected_indices=[], 
        nms_function=nms_function
    ))

    # 2) Single box
    boxes = np.array([[10, 10, 50, 50]], dtype=float)
    scores = np.array([0.9], dtype=float)
    results.append(run_nms_test(
        name="Test 2: single box",
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        expected_indices=[0],
        nms_function=nms_function
    ))

    # 3) No overlap -> keep all in descending score order
    boxes = np.array([
        [0, 0, 10, 10],      # idx 0, score 0.2
        [20, 20, 30, 30],    # idx 1, score 0.9
        [40, 40, 50, 50],    # idx 2, score 0.5
    ], dtype=float)
    scores = np.array([0.2, 0.9, 0.5], dtype=float)
    results.append(run_nms_test(
        name="Test 3: no overlap",
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        expected_indices=[1, 2, 0],
        nms_function=nms_function
    ))

    # 4) Strong overlap between first two boxes -> keep highest + separate box
    boxes = np.array([
        [10, 10, 50, 50],    # idx 0, high score
        [12, 12, 48, 48],    # idx 1, highly overlapping with idx 0
        [100, 100, 140, 140] # idx 2, separate
    ], dtype=float)
    scores = np.array([0.95, 0.80, 0.70], dtype=float)
    results.append(run_nms_test(
        name="Test 4: suppress highly-overlapping box",
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        expected_indices=[0, 2],
        nms_function=nms_function
    ))

    # 5) Chain overlap case (A overlaps B, B overlaps C, A weakly overlaps C)
    boxes = np.array([
        [0, 0, 20, 20],      # idx 0
        [5, 5, 25, 25],      # idx 1
        [10, 10, 30, 30],    # idx 2
    ], dtype=float)
    scores = np.array([0.90, 0.85, 0.80], dtype=float)
    results.append(run_nms_test(
        name="Test 5: chain overlap with threshold 0.3",
        boxes=boxes,
        scores=scores,
        iou_threshold=0.3,
        expected_indices=[0, 2],
        nms_function=nms_function
    ))

    # Show image if any test failed
    if not all(results):
        img_path = os.path.join(os.path.dirname(__file__), '..', 
                                '009_object_detection', 'youshallnotpass.png')
        if os.path.isfile(img_path):
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(mpimg.imread(img_path))
            ax.axis('off')
            ax.set_title(f'{sum(not r for r in results)} of {len(results)} tests failed',
                         fontsize=12, color='red', fontweight='bold')
            plt.tight_layout(); plt.show()