import omni.appwindow
import carb.input
from carb.input import KeyboardEventType
import omni.usd
from pxr import Usd, Sdf

class door_controller:
    def __init__(self, left_door, right_door, parent, offset=0.0, keyboard_input=carb.input.KeyboardInput.SPACE):
        self.stage = omni.usd.get_context().get_stage()        
        self.doors_open_state = False
        self._attrib_name = "drive:linear:physics:targetPosition"
        
        self.left_door_prim = self.find_prims_by_name_with_parent(left_door, parent)[0]
        print(self.left_door_prim.GetPath().pathString)
        self.right_door_prim = self.find_prims_by_name_with_parent(right_door, parent)[0]
        print(self.right_door_prim.GetPath().pathString)
        self.keyboard_input = keyboard_input
        self.offset = offset

    def toggle_doors(self):
        if self.doors_open_state:
            self.close_doors()
        else:
            self.open_doors()

    def open_doors(self):
        print("Opening doors")
        self.set_attr_value(self.left_door_prim, -self.offset)
        self.set_attr_value(self.right_door_prim, self.offset)
        self.doors_open_state = True

    def close_doors(self):
        print("Closing doors")
        self.set_attr_value(self.left_door_prim, 0)
        self.set_attr_value(self.right_door_prim, 0)
        self.doors_open_state = False

    def find_prims_by_name(self, prim_name):
        found_prims = [x for x in self.stage.Traverse() if x.GetName() == prim_name]
        return found_prims
    
    def find_prims_by_name_with_parent(self, prim_name, parent_name):
        found_prims = [x for x in self.stage.Traverse() if x.GetName() == prim_name]
        return_prims = []

        for prim in found_prims:
            prim_path = prim.GetPath().pathString
            if parent_name in prim_path:
                return_prims.append(prim)
        return return_prims
    
    def set_attr_value(self, prim, value):               
        attr = prim.GetAttribute(self._attrib_name)            
        attr.Set(value)


class door_manager:
    def __init__(self):
        self.controllers = {}
        
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._keyboard_sub_id = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_input)
    
    def __del__(self):
        if self._keyboard_sub_id:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub_id)
    
    def add_controller(self, key, controller):
        self.controllers[key] = controller
    
    def on_keyboard_input(self, e):
        if e.type == KeyboardEventType.KEY_PRESS:
            for key, controller in self.controllers.items():
                if e.input == controller.keyboard_input:
                    controller.toggle_doors()


if __name__ == "__main__":
    # Create a door manager
    manager = door_manager()
    
    number_keys = [carb.input.KeyboardInput.KEY_1, 
                   carb.input.KeyboardInput.KEY_2,
                   carb.input.KeyboardInput.KEY_3,
                   carb.input.KeyboardInput.KEY_4,
                   carb.input.KeyboardInput.KEY_5,
                   carb.input.KeyboardInput.KEY_6,
                   carb.input.KeyboardInput.KEY_7,
                   carb.input.KeyboardInput.KEY_8,
                   ]
    
    for i in range(8):
        door_outer = door_controller("Joint_SliderDoor_ElevatorDoorL", "Joint_SliderDoor_ElevatorDoorR", 
                                  f"ElevatorOuter_0{i+1}", offset=2, 
                                  keyboard_input=number_keys[i])
    
        door_inner = door_controller("Elevator_SliderDoor_ElevatorDoorR", "Elevator_SliderDoor_ElevatorDoorL", 
                                  f"ElevatorCabin_0{i+1}", offset=1.8, 
                                  keyboard_input=number_keys[i])

        manager.add_controller(f"inner_door{i+1}", door_inner)
        manager.add_controller(f"outer_door{i+1}", door_outer)

    # Create door controllers
    #door_01_inner = door_controller("Joint_SliderDoor_ElevatorDoorL", "Joint_SliderDoor_ElevatorDoorR",
    #                              "ElevatorOuter_01", offset=2,
    #                              keyboard_input=carb.input.KeyboardInput.KEY_1)
    
    #door_01_outer = door_controller("Elevator_SliderDoor_ElevatorDoorR", "Elevator_SliderDoor_ElevatorDoorL", 
    #                              "ElevatorCabin_01", offset=2, 
    #                              keyboard_input=carb.input.KeyboardInput.KEY_1)
    

    
    # Add controllers to manager
    #manager.add_controller("inner_door", door_01_inner)
    #manager.add_controller("outer_door", door_01_outer)