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
        this.terrain = new Terrain(new Vector3(0, 0, 0), 800);

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            torus: new defs.Torus(15, 15),
            torus2: new defs.Torus(3, 15),
            sphere: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 15),
            // TODO:  Fill in as many additional shape instances as needed in this key/value table.
            //        (Requirement 1)
            sphere1: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(1),
            sphere2: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(2),
            sphere3: new defs.Subdivision_Sphere(3),
            sphere4: new defs.Subdivision_Sphere(5),
            triangle: new defs.Triangle(),
            box: new defs.Cube(),
            "robot": new Shape_From_File("assets/wooden watch tower2.obj"),
            "stone": new Shape_From_File("assets/Cobblestones3/Files/untitled.obj"),
        };

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            ring: new Material(new Ring_Shader()),
            // TODO:  Fill in as many additional material objects as needed in this key/value table.
            //        (Requirement 4)
            room: new Material(new defs.Phong_Shader(),
                {ambient: 0.1, diffusivity: 0.8, specularity: 0.3, color: color(0.6,0.6,0.8,1)}),
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
            ambient: .3, diffusivity: 1, specularity: 1, texture: new Texture("assets/textures/Wood_Tower_Col.jpg")
        });
        this.stone = new Material(new defs.Bump_Map(1), {
            color: color(.5, .5, .5, 1),
            ambient: .3, diffusivity: 1, specularity: 1, texture: new Texture("assets/Cobblestones3/Textures/BrickRound0105_5_S.jpg"), 
            bump_texture: new Texture("assets/Cobblestones3/Textures/BrickRound0105_5_S_BUMP.png")
        });
        
        this.initial_camera_location = Mat4.look_at(vec3(0, 5, 20), vec3(0, 0, 0), vec3(0, 1, 0)).times(Mat4.translation(0, -5, -10, 1));
        
    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        this.key_triggered_button("View solar system", ["Control", "0"], () => this.attached = () => this.initial_camera_location);
        this.new_line();
        this.key_triggered_button("Attach to planet 1", ["Control", "1"], () => this.attached = () => this.planet_1);
        this.key_triggered_button("Attach to planet 2", ["Control", "2"], () => this.attached = () => this.planet_2);
        this.new_line();
        this.key_triggered_button("Attach to planet 3", ["Control", "3"], () => this.attached = () => this.planet_3);
        this.key_triggered_button("Attach to planet 4", ["Control", "4"], () => this.attached = () => this.planet_4);
        this.new_line();
        this.key_triggered_button("Attach to moon", ["Control", "m"], () => this.attached = () => this.moon);
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
        this.terrain.draw(context, program_state);

        // Lights
        const light_movement = 3*Math.cos(Math.PI*t/2);
        const light_movement2 = 3*Math.sin(Math.PI*t/2);
        const light_height = 8 + Math.sin(Math.PI*t);
        const light_color = color(1, 0.8, 0.7, 1);
        const light_intensity = 100000000;
        const light_position = vec4(light_movement, light_height, light_movement2, 1);

        program_state.lights = [new Light(light_position, light_color, light_intensity)];
        // program_state.current_terrain = this.terrain;

        const light_orb_transform = origin.times(Mat4.translation(light_movement, light_height, light_movement2, 1)).times(Mat4.scale(0.5, 0.5, 0.5));
        this.shapes.sphere4.draw(context, program_state, light_orb_transform, this.materials.light.override({color: light_color}));

        // // Room
        // let room_size = [10, 5, 10]; // W,H,D
        // let s_width = 0.5;

        // // Floor
        // let floor_transform = origin.times(Mat4.scale(room_size[0]+1, -s_width, room_size[2]+1));
        // this.shapes.box.draw(context, program_state, floor_transform, this.materials.room);

        // // Right Wall
        // let r_wall_transform = origin.times(Mat4.translation(-room_size[0]-s_width, room_size[1]+s_width, 0, 1)).times(Mat4.scale(s_width, room_size[1], room_size[2]+1));
        // this.shapes.box.draw(context, program_state, r_wall_transform, this.materials.room);

        // // Left Wall
        // let l_wall_transform = origin.times(Mat4.translation(room_size[0]+s_width, room_size[1]+s_width, 0, 1)).times(Mat4.scale(s_width, room_size[1], room_size[2]+1));
        // this.shapes.box.draw(context, program_state, l_wall_transform, this.materials.room);
        
        // // Back Wall
        // let b_wall_transform = origin.times(Mat4.translation(0, room_size[1]+s_width, -room_size[2]-s_width, 1)).times(Mat4.scale(room_size[0], room_size[1], s_width));
        // this.shapes.box.draw(context, program_state, b_wall_transform, this.goldy);

        // Sphere
        let sphere_transform1 = origin.times(Mat4.translation(-4, 4, -4, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform1, this.brick);
        
        let sphere_transform2 = origin.times(Mat4.translation(0, 3, -6, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform2, this.materials.sphere2);

        let sphere_transform3 = origin.times(Mat4.translation(6, 7, -2, 1));
        this.shapes.sphere4.draw(context, program_state, sphere_transform3, this.materials.sphere3);

        // Robot
        let robot_transform = origin.times(Mat4.scale(2, 2, 2)).times(Mat4.translation(0, 2.65, 0, 1));
        this.shapes.robot.draw(context, program_state, robot_transform, this.stone);

        // Stone
        let stone_transform = origin.times(Mat4.scale(2, 2, 2)).times(Mat4.translation(0, 1, 3, 1));
        this.shapes.stone.draw(context, program_state, stone_transform, this.stone);
        this.terrain.update(program_state)
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

class Ring_Shader extends Shader {
    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
        gl.uniform1f(gpu.time, material.time);
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));

        this.send_material(context, gpu_addresses, material);
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 point_position;
        varying vec4 center;
        uniform vec4 shape_color;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform float time;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        
        void main(){
            gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
            center = model_transform * vec4(0., 0., 0., 1.0);
            point_position = model_transform * vec4(position, smoothness);
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){
            float d = length(point_position.xyz - center.xyz);
            float v = (sin(d*20.) + 1.)/2.;
            gl_FragColor = shape_color*time;
        }`;
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

            if (material.texture && material.texture.ready) {
                // Select texture unit 0 for the fragment shader Sampler2D uniform called "texture":
                context.uniform1i(gpu_addresses.texture, 0);
                // For this draw, use the texture image from correct the GPU buffer:
                material.texture.activate(context);
            }
        }
    }

const Bump_Map = defs.Bump_Map =
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
                    vec4 tex_color = texture2D( texture, f_tex_coord );
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

    class TerrainShape extends Shape {
        constructor(size, height_map, max_height = 50) {
          super("position", "normal", "texture_coord");
          this.SIZE = size;
          this.MAX_HEIGHT = max_height;
          this.MAX_PIXEL_COLOR = 256 * 256 * 256;
      
          let VERTEX_COUNT = height_map.height;
      
          this.heights = new Array(VERTEX_COUNT);
          for (let i = 0; i < VERTEX_COUNT; i++) {
            this.heights[i] = new Array(VERTEX_COUNT);
          }
      
          let data = this.getImageData(height_map);
          for (let i = 0; i < VERTEX_COUNT; i++) {
            for (let j = 0; j < VERTEX_COUNT; j++) {
              let height = this.get_height(height_map, data, j, i);
      
              this.heights[j][i] = height;
              this.arrays.position.push(
                vec(
                  (-j / (VERTEX_COUNT - 1)) * this.SIZE,
                  height,
                  (-i / (VERTEX_COUNT - 1)) * this.SIZE
                )
              );
              let normal = this.calculate_normal(j, i, height_map, data);
      
              this.arrays.normal.push(vec(normal[0], normal[1], normal[2]));
              this.arrays.texture_coord.push(
                vec(j / VERTEX_COUNT - 1, i / VERTEX_COUNT - 1)
              );
            }
          }
          for (let gz = 0; gz < VERTEX_COUNT - 1; gz++) {
            for (let gx = 0; gx < VERTEX_COUNT - 1; gx++) {
              let top_left = gz * VERTEX_COUNT + gx;
              let top_right = top_left + 1;
              let bottom_left = (gz + 1) * VERTEX_COUNT + gx;
              let bottom_right = bottom_left + 1;
              this.indices.push(top_left);
              this.indices.push(bottom_left);
              this.indices.push(top_right);
              this.indices.push(top_right);
              this.indices.push(bottom_left);
              this.indices.push(bottom_right);
            }
          }
        }
      
        get_height(height_map, data, x, z) {
          if (x < 0 || x >= height_map.height || z < 0 || z >= height_map.height)
            return 0;
      
          let pixel = this.getPixel(data, x, z);
      
          let height = pixel.r * pixel.g * pixel.b;
          height -= this.MAX_PIXEL_COLOR / 2;
          height /= this.MAX_PIXEL_COLOR / 2;
          height *= this.MAX_HEIGHT;
      
          return height;
        }
      
        calculate_normal(x, z, height_map, data) {
          let height_left = this.get_height(height_map, data, x - 1, z);
          let height_right = this.get_height(height_map, data, x + 1, z);
          let height_down = this.get_height(height_map, data, x, z - 1);
          let height_up = this.get_height(height_map, data, x, z + 1);
      
          let normal = vec(
            height_left - height_right,
            2 * this.SIZE,
            height_down - height_up
          );
          normal.normalize();
          return normal;
        }
      
        getImageData(image) {
          var canvas = document.createElement("canvas");
          canvas.width = image.width;
          canvas.height = image.height;
      
          var context = canvas.getContext("2d");
          context.drawImage(image, 0, 0);
      
          return context.getImageData(0, 0, image.width, image.height);
        }
      
        getPixel(imagedata, x, y) {
          var position = (x + imagedata.width * y) * 4,
            data = imagedata.data;
          return {
            r: data[position],
            g: data[position + 1],
            b: data[position + 2],
            a: data[position + 3]
          };
        }
      }

class Terrain {
        constructor(position, size) {
          this.position = vec3(
            position[0] * size,
            position[1] * size,
            position[2] * size
          );
          this.size = size;
          this.height_map = new Image();
          this.height_map.src = "assets/heightmap.png";
          this.height_map.onload = () => {
            this.shape = new TerrainShape(size, this.height_map);
          };
          this.material = new Material(new TerrainShader(10), {
            texture: new Texture("assets/stars.png"),
            ambient: 0.8,
            diffusivity: 1.0,
            specularity: 0
          });
        }
      
        update(program_state) {
            this.material.ambient = Math.max(
              0.1,
              this.material.ambient - program_state.dt
            );
        }
      
        get_height(world_x, world_z) {
          if (!this.shape) return 0;
          let terrain_x = this.position[0] - world_x;
          let terrain_z = this.position[2] - world_z;
      
          let grid_square_size = this.size / (this.shape.heights.length - 1);
          let grid_x = Math.floor(terrain_x / grid_square_size);
          let grid_z = Math.floor(terrain_z / grid_square_size);
      
          if (
            grid_x >= this.shape.heights.length - 1 ||
            grid_z >= this.shape.heights.length - 1 ||
            grid_x < 0 ||
            grid_z < 0
          )
            return 0;
      
          let x_coord = (terrain_x % grid_square_size) / grid_square_size;
          let z_coord = (terrain_z % grid_square_size) / grid_square_size;
      
          var ans;
          if (x_coord <= 1 - z_coord) {
            ans = this.barry_centric(
                new Vector3(0, this.shape.heights[grid_x][grid_z], 0),
                new Vector3(1, this.shape.heights[grid_x + 1][grid_z], 0),
                new Vector3(0, this.shape.heights[grid_x][grid_z + 1], 1),
                new Vector3(x_coord, z_coord)
            );
          } else {
            ans = this.barry_centric(
                new Vector3(1, this.shape.heights[grid_x + 1][grid_z], 0),
                new Vector3(1, this.shape.heights[grid_x + 1][grid_z + 1], 1),
                new Vector3(0, this.shape.heights[grid_x][grid_z + 1], 1),
                new Vector3(x_coord, z_coord)
            );
          }
      
          return ans;
        }
      
        barry_centric(p1, p2, p3, pos) {
          let det =
            (p2[2] - p3[2]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[2] - p3[2]);
          let l1 =
            ((p2[2] - p3[2]) * (pos[0] - p3[0]) +
              (p3[0] - p2[0]) * (pos[1] - p3[2])) /
            det;
          let l2 =
            ((p3[2] - p1[2]) * (pos[0] - p3[0]) +
              (p1[0] - p3[0]) * (pos[1] - p3[2])) /
            det;
          let l3 = 1 - l1 - l2;
          return l1 * p1[1] + l2 * p2[1] + l3 * p3[1];
        }
      
        draw(context, program_state) {
        //   if (!this.shape) return;
          let terrain_transform = Mat4.identity();
          terrain_transform = terrain_transform.times(
            Mat4.translation([this.position[0], this.position[1], this.position[2]])
          );
          this.shape.draw(context, program_state, terrain_transform, this.material);
        }
}

class TerrainShader extends Textured_Phong {
    vertex_glsl_code() {
    // ********* VERTEX SHADER *********
    return (
        this.shared_glsl_code() +
        `
            varying vec2 f_tex_coord;
            attribute vec3 position, normal;                            // Position is expressed in object coordinates.
            attribute vec2 texture_coord;
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
            void main()
            {                                                                   // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                                                                                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                                                // Turn the per-vertex texture coordinate into an interpolated variable.
                f_tex_coord = texture_coord * 40.0;
                gl_Position = model_transform * vec4(position, 1.0);
            } `
    );
    }
}
