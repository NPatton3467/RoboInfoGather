import numpy as np
import openai

def get_map_params(obj_tp, map_original_size, map_original_resolution):
    """
    Calculates parameters needed for mapping form world to Belief coordinates using LLM and previous params

    :param: (obj_tp) Object type used to query LLM for new map resolution
    :param: (map_original_size) The original size of the the traversable map
    :param: (map_original_resolution) The original resolution of the traversable map

    :returns: a dict {'res' : map_resolution, 'og_res' : map_original_resolution, 'size' : map_size, 'og_size' : map_original_size}
    """

    # Query LLM for map resolution
    f = open('res_pre_prompt.txt', 'r')
    pre_prompt = f.read()
    f.close()
    prompt = pre_prompt + str(obj_tp) + '\n```\nOutput:\n```\n'

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{prompt}"}],
        stream=False,
        temperature=temp
    )

    map_resolution = float(sketch = response.choices[0].message.content)

    map_size = map_size = int(
                    map_original_size * map_original_resolution / map_resolution
                )

    return {'res' : map_resolution, 'og_res' : map_original_resolution, 'size' : map_size, 'og_size' : map_original_size}

def world_to_map(xy, map_resolution, map_size):
    return np.flip((np.array(xy) / map_resolution + map_size / 2.0)).astype(np.int)

def get_trav_map(maps_path, floor, resolution, og_resolution):
        """
        Loads the traversability map
        """

        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        map_size = None
        trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

        # If we do not initialize the original size of the traversability map, we obtain it from the image
        # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
        # original map size
        height, width = trav_map.shape
        map_default_resolution = og_resolution
        map_size = int(
            height * map_default_resolution / resolution
        )

        # We resize the traversability map to the new size computed before
        trav_map = cv2.resize(trav_map, (map_size, map_size))

        # We make the pixels of the image to be either 0 or 255
        trav_map[trav_map < 255] = 0

        return trav_map