# Segmentation

To run

```commandline
python main.py --input_path video_2.mp4 --output_path video_2_seg.mp4 --method contour
```

## Notes
I modified a solution found online. Not the best but I chose a lightweight, interpretable, tunable solution.

To allow the code base to easily incorporate new segmentation methods an abstract class was implemented.
