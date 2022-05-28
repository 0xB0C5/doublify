# Doublify
## Features
- Converts 4 panel charts to 8 panel.
- Can batch process recursive folders of songs.
- Removes old autogen charts so you can re-doublify charts when a new version comes out.
- Skips charts which have non-autogen double charts so you can doublify your whole Songs folder and you won't lose anything.
- Generates patterns suitable for stamina & footspeed.

## Issues
- Quads & triples will be converted to jumps.
- Holds & rolls are included but don't affect pattern generation, so this can lead to double steps.
- Footswitches will be converted to jacks.
- Does not distinguish between jumps & 1-foot brackets. All will be converted to jumps.
- An all-jump section will not move across the pads.
- Jump patterns are not ideal.

## Usage
- Currently only Windows is supported.
- Install python 3.8: https://www.python.org/downloads/release/python-380/
- In File Explorer, drag a folder or simfile onto doublify.bat
- Wait for the message "Press any key to continue . . ." to appear. This may take a while if there are lots of songs.
- Press any key
