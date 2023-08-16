import subprocess
from pathlib import Path

this_dir = Path(__file__).parents[0]
input_dir = this_dir.parents[2] / 'tumbling_temp' / 'flow_sequence'
temp_dir = this_dir
output_dir = this_dir

# Animate the frames into a gif:
palette = temp_dir / "palette.png"
filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
print("Converting pngs to gif...", end=' ')
convert_command_1 = [
    '/Users/julia/Desktop/ffmpeg',
    '-f', 'image2',
    '-i', input_dir / ('%03d.png'),
    '-vf', filters + ",palettegen",
    '-y', palette]
convert_command_2 = [
    '/Users/julia/Desktop/ffmpeg',
    '-framerate', '20',
    '-f', 'image2',
    '-i', input_dir / ('%03d.png'),
    '-i', palette,
    '-lavfi', filters + " [x]; [x][1:v] paletteuse",
    '-y', output_dir / '03_flow_cytometry.gif']
for convert_command in convert_command_1, convert_command_2:
    try:
        with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
            f.write("So far, everthing's fine...\n")
            f.flush()
            subprocess.check_call(convert_command, stderr=f, stdout=f)
            f.flush()
        (temp_dir / 'conversion_messages.txt').unlink()
    except: # This is unlikely to be platform independent :D
        print("GIF conversion failed. Is ffmpeg installed?")
        raise
print('done.')
