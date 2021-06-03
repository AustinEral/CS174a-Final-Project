# CS 174a Final Project

Francesco Colonnese & Austin Eral

## Team Stone

Our project showcases several forms of texture mapping. We use Bump Mapping to create dynamic lighting effects on objects and create textures that change over time to simulate fluid dynamics and 2-dimensional animations within our scene.

## Advanced Features
### Bump Mapping
Our bump mapping works by using the original texture that is mapped to an object using a Phong Texture Shader to translate the normals of pixels in the fragment shaders. This gives the illusion that the object is multi-poly and not flat.

### Animated Textures
By creating an array of textures for an object, we can reassign textures to the object over time. We split videos and gif animations into their frames for each texture since glsl does not support animated gif texture rendering.

## Explore
### Fountain
The fountain uses both advanced features to simulate the reflectivity of moving water and how its surface appears to change over time.

### Marble Floor
The marble floor uses a bump mapped texture that is rescaled over the object to create the appearance of more tiles.

### Sky sphere (box)
The sky is a large sphere with a circular panoramic image mapped to the object. This avoids the visible issue where the texture meets the other side of itself. The sphere is not moving but the texture is rotating across the sphere.