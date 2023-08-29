function jump_to_009() {
  var video = document.getElementById("tumbling_animation_video");
  video.currentTime = 9;
}

function jump_to_067() {
  var video = document.getElementById("tumbling_animation_video");
  video.currentTime = 67;
}

function jump_to_101() {
  var video = document.getElementById("tumbling_animation_video");
  video.currentTime = 101;
}

function jump_to_162() {
  var video = document.getElementById("tumbling_animation_video");
  video.currentTime = 162;
}

function update_general_tumbling_figure() {
  var filename = "./figures/02_generic_tumbling/" + document.getElementById("general_tumbling_selector").value + "_relaxation/05_tumbling.gif";
  var image = document.getElementById("general_tumbling");
  image.src = filename;
  }  

function update_power_variation_figure() {
  var filename = "./figures/03_photobleaching_photoswitching/" + document.getElementById("power_variation_selector").value + "_minimal_hardware/04_" + document.getElementById("power_variation_selector").value + ".gif";
  var image = document.getElementById("power_variation");
  image.src = filename;
  }

function update_triplets_camera_figure() {
  var filename = "./figures/04_triplets_camera_protein_complexes/" + document.getElementById("triplets_camera_selector").value + "/05_triplets_camera.gif";
  var image = document.getElementById("triplets_camera");
  image.src = filename;
  } 

function update_triplets_flow_figure() {
  var filename = "./figures/05_flow_cytometry_triplets/" + document.getElementById("triplets_flow_selector").value + ".gif";
  var image = document.getElementById("triplets_flow");
  image.src = filename;
  }      

function update_sp8_tumbling_figure() {
  var filename = "./figures/06_triplets_SP8_confocal/simulation_expt_beads_" + document.getElementById("sp8_tumbling_selector").value + ".png";
  var image = document.getElementById("sp8_tumbling");
  image.src = filename;
  }    

function update_tcspc_figure() {
  var filename = "./figures/SI_tcspc_of_OADF_signal/" + document.getElementById("FP_tcspc_selector").value + "_triplet_tcspc/08_prompt_triggered_decays_with_fits.png"; 
  var image = document.getElementById("tcspc_fit");
  image.src = filename;
  }  

function update_tcspc_overlay_figure() {
  var filename = "./figures/SI_tcspc_of_OADF_signal/" + document.getElementById("FP_tcspc_overlaid_selector").value + "_triplet_tcspc/07_prompt_triggered_tcspc_overlaid.png"; 
  var image = document.getElementById("tcspc_overlaid");
  image.src = filename;
  } 

function update_SP8_schematic_figure() {
  var filename = "./figures/SI_pulse_setup_and_ctrls/" + document.getElementById("SP8_schematic_selector").value + "_pulse_setup.png"; 
  var image = document.getElementById("SP8_schematic");
  image.src = filename;
  } 