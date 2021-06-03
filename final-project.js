import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene, Texture,
} = tiny;

export class Shape_From_File extends Shape {                                   // **Shape_From_File** is a versatile standalone Shape that imports
                                                                               // all its arrays' data from an .obj 3D model file.
    constructor(filename) {
        super("position", "normal", "texture_coord");
        // Begin downloading the mesh. Once that completes, return
        // control to our parse_into_mesh function.
        this.load_file(filename);
    }

    load_file(filename) {                             // Request the external file and wait for it to load.
        // Failure mode:  Loads an empty shape.
        return fetch(filename)
            .then(response => {
                if (response.ok) return Promise.resolve(response.text())
                else return Promise.reject(response.status)
            })
            .then(obj_file_contents => this.parse_into_mesh(obj_file_contents))
            .catch(error => {
                this.copy_onto_graphics_card(this.gl);
            })
    }

    parse_into_mesh(data) {                           // Adapted from the "webgl-obj-loader.js" library found online:
        var verts = [], vertNormals = [], textures = [], unpacked = {};

        unpacked.verts = [];
        unpacked.norms = [];
        unpacked.textures = [];
        unpacked.hashindices = {};
        unpacked.indices = [];
        unpacked.index = 0;

        var lines = data.split('\n');

        var VERTEX_RE = /^v\s/;
        var NORMAL_RE = /^vn\s/;
        var TEXTURE_RE = /^vt\s/;
        var FACE_RE = /^f\s/;
        var WHITESPACE_RE = /\s+/;

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            var elements = line.split(WHITESPACE_RE);
            elements.shift();

            if (VERTEX_RE.test(line)) verts.push.apply(verts, elements);
            else if (NORMAL_RE.test(line)) vertNormals.push.apply(vertNormals, elements);
            else if (TEXTURE_RE.test(line)) textures.push.apply(textures, elements);
            else if (FACE_RE.test(line)) {
                var quad = false;
                for (var j = 0, eleLen = elements.length; j < eleLen; j++) {
                    if (j === 3 && !quad) {
                        j = 2;
                        quad = true;
                    }
                    if (elements[j] in unpacked.hashindices)
                        unpacked.indices.push(unpacked.hashindices[elements[j]]);
                    else {
                        var vertex = elements[j].split('/');

                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 0]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 1]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 2]);

                        if (textures.length) {
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 0]);
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 1]);
                        }

                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 0]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 1]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 2]);

                        unpacked.hashindices[elements[j]] = unpacked.index;
                        unpacked.indices.push(unpacked.index);
                        unpacked.index += 1;
                    }
                    if (j === 3 && quad) unpacked.indices.push(unpacked.hashindices[elements[0]]);
                }
            }
        }
        {
            const {verts, norms, textures} = unpacked;
            for (var j = 0; j < verts.length / 3; j++) {
                this.arrays.position.push(vec3(verts[3 * j], verts[3 * j + 1], verts[3 * j + 2]));
                this.arrays.normal.push(vec3(norms[3 * j], norms[3 * j + 1], norms[3 * j + 2]));
                this.arrays.texture_coord.push(vec(textures[2 * j], textures[2 * j + 1]));
            }
            this.indices = unpacked.indices;
        }
        this.normalize_positions(false);
        this.ready = true;
    }

    draw(context, program_state, model_transform, material) {               // draw(): Same as always for shapes, but cancel all
        // attempts to draw the shape before it loads:
        if (this.ready)
            super.draw(context, program_state, model_transform, material);
    }
}

export class Final_Project extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            torus: new defs.Torus(15, 15),
            torus2: new defs.Torus(3, 15),
            sphere: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 40),
            sphere4: new defs.Subdivision_Sphere(5),
            triangle: new defs.Triangle(),
            box: new defs.Cube(),
            "robot": new Shape_From_File("assets/Atlas.obj"),
            "stone": new Shape_From_File("assets/Cobblestones3/Files/untitled.obj"),
            "fountain": new Shape_From_File("assets/fountain2/fountain.obj")
        };

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            room: new Material(new defs.Textured_Phong(),
                {            color: color(0, 0, 0, 1),
                    ambient: 0.8, diffusivity: 1, specularity: 1, texture: new Texture("assets/ffff.png")}),
            sphere1: new Material(new defs.Phong_Shader(),
                {ambient: 0.2, diffusivity: 1, specularity: 0.5, color: color(1,0.7,0,1), smoothness: 40, time: 0}),
            sphere2: new Material(new defs.Phong_Shader(),
                {ambient: 0.2, diffusivity: 1, specularity: 0.5, color: color(0,0.8,0.8,1)}),
            sphere3: new Material(new defs.Phong_Shader(),
                {ambient: 0.2, diffusivity: 1, specularity: 0.5, color: color(0.8,0.2,0.4,1)}),
            light: new Material(new defs.Phong_Shader(),
                {ambient: 1, diffusivity: 0, specularity: 0, color: color(1,1,1,1)}),
        }
        this.goldy = new Material(new defs.Fake_Bump_Map(1), {
            color: color(.5, .5, .5, 1),
            ambient: .3, diffusivity: 1, specularity: 1, texture: new Texture("assets/goldy.png")
        });
        this.brick = new Material(new defs.Fake_Bump_Map(1), {
            color: color(.5, .5, .5, 1),
            ambient: .3, diffusivity: 1, specularity: 1, texture: new Texture("assets/brick.png")
        });
        this.bumpy = new Material(new defs.Fake_Bump_Map(1), {
            color: color(.5, .5, .5, 1),
            ambient: .3, diffusivity: 1, specularity: 1, texture: new Texture("assets/textures/bumpy.png")
        });
        this.stone = new Material(new defs.Fake_Bump_Map(1), {
            color: color(0, 0, 0, 1),
            ambient: 1, diffusivity: 1, specularity: 1, texture: new Texture("assets/woodfloor.jpg"), 
            bump_texture: new Texture("assets/Cobblestones3/Textures/BrickRound0105_5_S_BUMP.png")
        });
        this.tile = new Material(new defs.Bump_Map_Texure_x4(2), {
            color: color(0, 0, 0, 1),
            ambient: 0.8, diffusivity: 1, specularity: 1, texture: new Texture("assets/tile.jpg")
        });
        this.sky = new Material(new Texture_Scroll_X(), {
          color: color(0, 0, 0, 1),
          ambient: 1, diffusivity: 1, specularity: 1, texture: new Texture("assets/invert.png"),
        });
        this.num_water_frames = 16;
        this.water_textures = [this.num_water_frames];
        for (let i = 0; i < this.num_water_frames; i++) {
            this.water_textures[i] = new Texture("assets/water/" + i + ".gif");
        }
        this.water = new Material(new defs.Fake_Bump_Map(2), {
            color: color(0, 0, 0, 0.8),
            ambient: 0.8, diffusivity: 1, specularity: 1, texture: this.water_textures[0], 
        });

        this.num_ucla_frames = 110;
        this.ucla_textures = [this.num_ucla_frames];
        for (let i = 0; i < this.num_ucla_frames; i++) {
            try {
                const actual = i + 1;
                const str = "assets/ucla/" + actual + "-crop.jpg"
                this.ucla_textures[i] = new Texture(str);
            }
            catch (error) {
                console.log('uh oh')
            }
        }
        this.ucla = new Material(new defs.Fake_Bump_Map(2), {
            color: color(0, 0, 0, 0.8),
            ambient: 0.8, diffusivity: 1, specularity: 1, texture: this.ucla_textures[0], 
        });
        this.fountain = new Material(new defs.Fake_Bump_Map(2), {
            color: color(0, 0, 0, 1),
            ambient: .3, diffusivity: 0.5, specularity: 1, texture: new Texture("assets/fountain/fountain.png"),
        });

        this.initial_camera_location = Mat4.look_at(vec3(0, 0, 0), vec3(10, 0, 0), vec3(0, 5, 0)).times(Mat4.translation(20, -8, 0, 1));

    }

    make_control_panel() {
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }

        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);
        
        let origin = Mat4.identity();

        // Time
        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;
        const yellow = hex_color("#fac91a");
        
        // Lights
        const light_movement = 3*Math.cos(Math.PI*t/2);
        const light_movement2 = 3*Math.sin(Math.PI*t/2);
        const light_height = 8 + Math.sin(Math.PI*t);
        const light_color = color(1, 0.8, 0.7, 1);
        const light_intensity = 1000;
        const light_position = vec4(light_movement, light_height, light_movement2, 1);

        const fountain_light_position = vec4(30, 5, 30, 1);

        program_state.lights = [new Light(light_position, light_color, light_intensity), new Light(fountain_light_position, light_color, 100)];
        const light_orb_transform2 = origin.times(Mat4.translation(30, 4, 30, 1)).times(Mat4.translation(30, 4, 30, 1)).times(Mat4.scale(1, 1, 1));

        const light_orb_transform = origin.times(Mat4.translation(light_movement, light_height, light_movement2, 1)).times(Mat4.scale(0.5, 0.5, 0.5));
        this.shapes.sphere4.draw(context, program_state, light_orb_transform, this.materials.light.override({color: light_color}));

        // Room
        let room_size = [40, 10, 40]; // W,H,D
        let s_width = 0.5;

        // Floor
        let floor_transform = origin.times(Mat4.scale(room_size[0]+1, -s_width, room_size[2]+1));
        this.shapes.box.draw(context, program_state, floor_transform, this.tile);

        // Right Wall
        let r_wall_transform = origin.times(Mat4.translation(-room_size[0]-s_width, room_size[1]+s_width, 0, 1)).times(Mat4.scale(s_width, room_size[1], room_size[2]+1));
        this.shapes.box.draw(context, program_state, r_wall_transform, this.materials.room);

        // Left Wall
        let l_wall_transform = origin.times(Mat4.translation(room_size[0]+s_width, room_size[1]+s_width, 0, 1)).times(Mat4.scale(s_width, room_size[1], room_size[2]+1));
        this.shapes.box.draw(context, program_state, l_wall_transform, this.materials.room);
        
        // Back Wall
        let b_wall_transform = origin.times(Mat4.translation(0, room_size[1]+s_width, -room_size[2]-s_width, 1)).times(Mat4.scale(room_size[0], room_size[1], s_width));
        this.shapes.box.draw(context, program_state, b_wall_transform, this.goldy);

        // Front Wall
        let f_wall_transform = origin.times(Mat4.translation(0, room_size[1]+s_width, room_size[2]+s_width, 1)).times(Mat4.scale(room_size[0], room_size[1], s_width));
        this.shapes.box.draw(context, program_state, f_wall_transform, this.materials.room);

        // Sphere
        let sphere_transform1 = origin.times(Mat4.translation(-5, 4, -4, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform1, this.brick);
        
        let sphere_transform2 = origin.times(Mat4.translation(0, 9, -6, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform2, this.materials.sphere2);

        let sphere_transform3 = origin.times(Mat4.translation(6, 7, -2, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform3, this.materials.sphere3);

        

        // Stone
        let stone_transform = origin.times(Mat4.scale(2, 2, 2)).times(Mat4.translation(3, 0.3, 3, 1));
        this.shapes.stone.draw(context, program_state, stone_transform, this.stone);

        //Sky
        let sky_transform = origin.times(Mat4.scale(500, 500, 500));
        this.shapes.sphere4.draw(context, program_state, sky_transform, this.sky);
        
        // Stand
        let stand_transform = origin.times(Mat4.scale(3, 2, 3).times(Mat4.translation(10, 0.3, 10, 1)));
        // this.shapes.box.draw(context, program_state, stand_transform, this.stone);

        // Stand
        let stand_transform2 = origin.times(Mat4.scale(3, 2, 3).times(Mat4.translation(-10, 0.3, 10, 1)));
        this.shapes.box.draw(context, program_state, stand_transform2, this.stone);

        // Stand
        let stand_transform3 = origin.times(Mat4.scale(3, 2, 3).times(Mat4.translation(-10, 0.3, -10, 1)));
        this.shapes.box.draw(context, program_state, stand_transform3, this.stone);

        // Stand
        let stand_transform4 = origin.times(Mat4.scale(3, 2, 3).times(Mat4.translation(10, 0.3, -10, 1)));
        this.shapes.box.draw(context, program_state, stand_transform4, this.stone);

        // Water
        let water_transform1 = origin.times(Mat4.translation(30, 2.5, 30, 1)).times(Mat4.rotation(Math.PI/2, 0, 0, 1)).times(Mat4.scale(0.1, 6, 6));
        let water_transform2 = origin.times(Mat4.translation(30, 6.5, 30, 1)).times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.scale(2, 2, 2));
        let water_frame_rate = 0.05;
        this.shapes.box.draw(context, program_state, water_transform1, this.water.override({texture: this.water_textures[Math.floor(t/water_frame_rate % this.num_water_frames)]}));
        this.shapes.circle.draw(context, program_state, water_transform2, this.water.override({texture: this.water_textures[Math.floor(t/water_frame_rate % this.num_water_frames)]}));

        // Fountain
        let fountain_transform = origin.times(Mat4.translation(30, 5.5, 30, 1)).times(Mat4.rotation(-Math.PI/2, 1, 0, 0)).times(Mat4.scale(4, 4, 4));
        this.shapes.fountain.draw(context, program_state, fountain_transform, this.brick);
        let f1_transform = origin.times(Mat4.translation(23, 1.5, 23, 1)).times(Mat4.scale(1, 1.5, 1))
        for (let i = 0; i < 8; i++) {
            this.shapes.box.draw(context, program_state, f1_transform, this.brick);
            f1_transform = f1_transform.times(Mat4.translation(0, 0, 14, 1));
            this.shapes.box.draw(context, program_state, f1_transform, this.brick);
            f1_transform = f1_transform.times(Mat4.translation(2, 0, -14, 1));
        }
        f1_transform = f1_transform.times(Mat4.translation(-2, 0, 2, 1));
        for (let i = 0; i < 6; i++) {
            this.shapes.box.draw(context, program_state, f1_transform, this.brick);
            f1_transform = f1_transform.times(Mat4.translation(-14, 0, 0, 1));
            this.shapes.box.draw(context, program_state, f1_transform, this.brick);
            f1_transform = f1_transform.times(Mat4.translation(14, 0, 2, 1));
        }

        // Ucla
        let ucla_transform = origin.times(
            Mat4.translation(-30, room_size[1]/2-1.2, +10.45-room_size[2]-s_width, 1)).times(Mat4.rotation(0, -Math.PI / 2, 0, 1)).times(Mat4.scale(0.1, 3, 3));
        let ucla_frame_rate = 0.90;
        const iter = Math.floor(t/ucla_frame_rate % this.num_ucla_frames)
        this.shapes.box.draw(
            context, 
            program_state, 
            ucla_transform, 
            this.ucla.override({texture: this.ucla_textures[iter]}));

        // Robot
        let robot_transform = origin.times(Mat4.scale(2, 2, 2)).times(Mat4.translation(15, 4, -15, 1));
        this.shapes.robot.draw(context, program_state, robot_transform, this.bumpy);


        if (this.attached != undefined) {
            let desired = Mat4.inverse(this.attached().times(Mat4.translation(0, 0, 5)));
            if (this.attached() == this.initial_camera_location) {
                desired = this.initial_camera_location;
            }
            let blending_factor = 0.1;
            program_state.camera_inverse = desired.map((x,i) => Vector.from(program_state.camera_inverse[i]).mix(x, blending_factor));
        }

    }
}

class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;

        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, vertex_worldspace;
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }

    vertex_glsl_code() {
        let rand = Math.random();
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;

            

            varying vec3 color;
            void main(){
                gl_Position = projection_camera_model_transform * vec4( position, `+ 1.0 + ` );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                color = phong_model_lights( normalize( N ), vertex_worldspace );
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            varying vec3 color;
            void main(){                                                           
                // Compute an initial (ambient) color:
                gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                // Compute the final color with contributions from lights:
                gl_FragColor.xyz += color;
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

const Phong_Shader = defs.Phong_Shader =
    class Phong_Shader extends Shader {
        // **Phong_Shader** is a subclass of Shader, which stores and manages a GPU program.
        // Graphic cards prior to year 2000 had shaders like this one hard-coded into them
        // instead of customizable shaders.  "Phong-Blinn" Shading here is a process of
        // determining brightness of pixels via vector math.  It compares the normal vector
        // at that pixel with the vectors toward the camera and light sources.


        constructor(num_lights = 2) {
            super();
            this.num_lights = num_lights;
        }

        shared_glsl_code() {
            // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
            return ` precision mediump float;
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec4 shape_color;
                uniform vec3 squared_scale, camera_center;
        
                // Specifier "varying" means a variable's final value will be passed from the vertex shader
                // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
                // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
                varying vec3 N, vertex_worldspace;
                // ***** PHONG SHADING HAPPENS HERE: *****                                       
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
                    // phong_model_lights():  Add up the lights' contributions.
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                        // light will appear directional (uniform direction from all points), and we 
                        // simply obtain a vector towards the light by directly using the stored value.
                        // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                        // the point light's location from the current surface point.  In either case, 
                        // fade (attenuate) the light as the vector needed to reach it gets longer.  
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        // Compute the diffuse and specular components from the Phong
                        // Reflection Model, using Blinn's "halfway vector" method:
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  } `;
        }

        vertex_glsl_code() {
            // ********* VERTEX SHADER *********
            return this.shared_glsl_code() + `
                attribute vec3 position, normal;                            
                // Position is expressed in object coordinates.
                
                uniform mat4 model_transform;
                uniform mat4 projection_camera_model_transform;
        
                void main(){                                                                   
                    // The vertex's final resting place (in NDCS):
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    // The final normal vector in screen space.
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                  } `;
        }

        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            // A fragment is a pixel that's overlapped by the current triangle.
            // Fragments affect the final image or get discarded due to depth.
            return this.shared_glsl_code() + `
                void main(){                                                           
                    // Compute an initial (ambient) color:
                    gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                    // Compute the final color with contributions from lights:
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                  } `;
        }

        send_material(gl, gpu, material) {
            // send_material(): Send the desired shape-wide material qualities to the
            // graphics card, where they will tweak the Phong lighting formula.
            gl.uniform4fv(gpu.shape_color, material.color);
            gl.uniform1f(gpu.ambient, material.ambient);
            gl.uniform1f(gpu.diffusivity, material.diffusivity);
            gl.uniform1f(gpu.specularity, material.specularity);
            gl.uniform1f(gpu.smoothness, material.smoothness);
        }

        send_gpu_state(gl, gpu, gpu_state, model_transform) {
            // send_gpu_state():  Send the state of our whole drawing context to the GPU.
            const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
            gl.uniform3fv(gpu.camera_center, camera_center);
            // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
            const squared_scale = model_transform.reduce(
                (acc, r) => {
                    return acc.plus(vec4(...r).times_pairwise(r))
                }, vec4(0, 0, 0, 0)).to3();
            gl.uniform3fv(gpu.squared_scale, squared_scale);
            // Send the current matrices to the shader.  Go ahead and pre-compute
            // the products we'll need of the of the three special matrices and just
            // cache and send those.  They will be the same throughout this draw
            // call, and thus across each instance of the vertex shader.
            // Transpose them since the GPU expects matrices as column-major arrays.
            const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
            gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
            gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

            // Omitting lights will show only the material color, scaled by the ambient term:
            if (!gpu_state.lights.length)
                return;

            const light_positions_flattened = [], light_colors_flattened = [];
            for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
                light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
                light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
            }
            gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
            gl.uniform4fv(gpu.light_colors, light_colors_flattened);
            gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
        }

        update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
            // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
            // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
            // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
            // program (which we call the "Program_State").  Send both a material and a program state to the shaders
            // within this function, one data field at a time, to fully initialize the shader for a draw.

            // Fill in any missing fields in the Material object with custom defaults for this shader:
            const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
            material = Object.assign({}, defaults, material);

            this.send_material(context, gpu_addresses, material);
            this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
        }
    }


    const Textured_Phong = defs.Textured_Phong =
    class Textured_Phong extends Phong_Shader {
        // **Textured_Phong** is a Phong Shader extended to addditionally decal a
        // texture image over the drawn shape, lined up according to the texture
        // coordinates that are stored at each shape vertex.
        vertex_glsl_code() {
            // ********* VERTEX SHADER *********
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                attribute vec3 position, normal;                            
                // Position is expressed in object coordinates.
                attribute vec2 texture_coord;
                
                uniform mat4 model_transform;
                uniform mat4 projection_camera_model_transform;
        
                void main(){                                                                   
                    // The vertex's final resting place (in NDCS):
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    // The final normal vector in screen space.
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                    // Turn the per-vertex texture coordinate into an interpolated variable.
                    f_tex_coord = texture_coord;
                  } `;
        }

        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            // A fragment is a pixel that's overlapped by the current triangle.
            // Fragments affect the final image or get discarded due to depth.
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                uniform sampler2D texture;
                uniform float animation_time;
                
                void main(){
                    // Sample the texture image in the correct place:
                    vec4 tex_color = texture2D( texture, f_tex_coord );
                    if( tex_color.w < .01 ) discard;
                                                                             // Compute an initial (ambient) color:
                    gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                             // Compute the final color with contributions from lights:
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                  } `;
        }

        update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
            // update_GPU(): Add a little more to the base class's version of this method.
            super.update_GPU(context, gpu_addresses, gpu_state, model_transform, material);
            // Updated for assignment 4
            context.uniform1f(gpu_addresses.animation_time, gpu_state.animation_time / 1000);
            if (material.texture && material.texture.ready) {
                // Select texture unit 0 for the fragment shader Sampler2D uniform called "texture":
                context.uniform1i(gpu_addresses.texture, 0);
                // For this draw, use the texture image from correct the GPU buffer:
                material.texture.activate(context);
            }
        }
    }


const Bump_Map_Texure_x4 = defs.Bump_Map_Texure_x4 =
    class Bump_Map extends Textured_Phong {
        // **Fake_Bump_Map** Same as Phong_Shader, except adds a line of code to
        // compute a new normal vector, perturbed according to texture color.
        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                uniform sampler2D texture;
                uniform sampler2D bump_texture;
        
                void main(){
                    // Sample the texture image in the correct place:
                    vec2 new_coord = vec2(f_tex_coord.x*4.0, f_tex_coord.y*4.0);
                    vec4 tex_color = texture2D( texture, new_coord );
                    if( tex_color.w < .01 ) discard;
                    // Slightly disturb normals based on sampling the same image that was used for texturing:
                    vec3 bumped_N  = N + tex_color.rgb - .5*vec3(1,1,1);
                    // Compute an initial (ambient) color:
                    gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                    // Compute the final color with contributions from lights:
                    gl_FragColor.xyz += phong_model_lights( normalize( bumped_N ), vertex_worldspace );
                  } `;
        }
    }

    class Texture_Rotate extends Textured_Phong {
      // TODO:  Modify the shader below (right now it's just the same fragment shader as Textured_Phong) for requirement #7.
      fragment_glsl_code() {
          return this.shared_glsl_code() + `
              varying vec2 f_tex_coord;
              uniform sampler2D texture;
              uniform float animation_time;
              void main(){
                  // Sample the texture image in the correct place:
                  float t = animation_time * 2.0 * 3.14159 / 4.0;
                  // Modulo the angle so values don't grow forever
                  t = t - (2.0 * 3.1415 * floor(t / (2.0 * 3.14159)));
                  float x = f_tex_coord.x - 0.5;
                  float y = f_tex_coord.y - 0.5;
                  vec2 new_coord = vec2(sin(t) * x - cos(t) * y + 0.5, cos(t) * x + sin(t) * y + 0.5);
                  vec4 tex_color = texture2D( texture, new_coord );
                  if( tex_color.w < .01 ) discard;
                                                                           // Compute an initial (ambient) color:
                  gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                           // Compute the final color with contributions from lights:
                  gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
          } `;
      }
  }

  class Texture_Scroll_X extends Textured_Phong {
    // TODO:  Modify the shader below (right now it's just the same fragment shader as Textured_Phong) for requirement #6.
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
            varying vec2 f_tex_coord;
            uniform sampler2D texture;
            uniform float animation_time;
            
            void main(){
                // Sample the texture image in the correct place:
                // Modulo the texture coordinates so vec2 values don't grow forever
                vec2 new_coord = vec2(f_tex_coord.x - (animation_time - (100.0 * floor(animation_time / 100.0))) * 0.01, f_tex_coord.y);
                vec4 tex_color = texture2D( texture, new_coord);
                if( tex_color.w < .01 ) discard;
                                                                         // Compute an initial (ambient) color:
                gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                         // Compute the final color with contributions from lights:
                gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
        } `;
    }
}