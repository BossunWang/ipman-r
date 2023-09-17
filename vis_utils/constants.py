ESSENTIALS_DIR = 'data/essentials'
SMPLX_VERT_SEG = 'data/meshscripts/smplx_vert_segmentation.json'
SMPL_VERT_SEG = 'data/meshscripts/smpl_vert_segmentation.json'
SEGMENTS_DIR = 'data/essentials/yogi_segments'

frame_select_dict_combined = {
    'Akarna_Dhanurasana-a': 1226,
    'Akarna_Dhanurasana-b': 1126,
    'Akarna_Dhanurasana-c': 730,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-a': 872,
    'Boat_Pose_or_Paripurna_Navasana_-a': 486,
    'Boat_Pose_or_Paripurna_Navasana_-b': 392,
    'Boat_Pose_or_Paripurna_Navasana_-c': 340,
    'Boat_Pose_or_Paripurna_Navasana_-d': 300,
    'Boat_Pose_or_Paripurna_Navasana_-e': 314,
    'Boat_Pose_or_Paripurna_Navasana_-f': 280,
    'Bound_Angle_Pose_or_Baddha_Konasana_-a': 622,
    'Bound_Angle_Pose_or_Baddha_Konasana_-b': 510,
    'Bound_Angle_Pose_or_Baddha_Konasana_-c': 462,
    'Bound_Angle_Pose_or_Baddha_Konasana_-f': 390,
    'Bound_Angle_Pose_or_Baddha_Konasana_-g': 274,
    'Camel_Pose_or_Ustrasana_-c': 430,
    'Camel_Pose_or_Ustrasana_-d': 388,
    'Cat_Cow_Pose_or_Marjaryasana_-a': 326,
    'Cat_Cow_Pose_or_Marjaryasana_-b': 234,
    'Chair_Pose_or_Utkatasana_-a': 266,
    'Chair_Pose_or_Utkatasana_-b': 272,
    'Chair_Pose_or_Utkatasana_-c': 368,
    'Child_Pose_or_Balasana_-a': 468,
    'Cobra_Pose_or_Bhujangasana_-a': 594,
    'Cobra_Pose_or_Bhujangasana_-b': 470,
    'Cobra_Pose_or_Bhujangasana_-c': 444,
    'Cobra_Pose_or_Bhujangasana_-d': 476,
    'Corpse_Pose_or_Savasana_-a': 376,
    'Cow_Face_Pose_or_Gomukhasana_-a': 990,
    'Cow_Face_Pose_or_Gomukhasana_-b': 650,
    'Cow_Face_Pose_or_Gomukhasana_-c': 628,
    'Cow_Face_Pose_or_Gomukhasana_-e': 578,
    'Crane_(Crow)_Pose_or_Bakasana_-a': 848,
    'Crane_(Crow)_Pose_or_Bakasana_-b': 574,
    'Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a': 468,
    'Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a': 416,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a': 432,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-b': 380,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-c': 340,
    'Eagle_Pose_or_Garudasana_-a': 580,
    'Eight-Angle_Pose_or_Astavakrasana_-a': 686,
    'Extended_Puppy_Pose_or_Uttana_Shishosana_-a': 464,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-a': 590,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-b': 538,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-d': 634,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-e': 510,
    'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-a': 496,
    'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-b': 572,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a': 1216,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-b': 752,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c': 1546,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-d': 766,
    'Firefly_Pose_or_Tittibhasana_-a': 644,
    'Firefly_Pose_or_Tittibhasana_-b': 496,
    'Fish_Pose_or_Matsyasana_-a': 1286,
    'Fish_Pose_or_Matsyasana_-c': 1160,
    'Fish_Pose_or_Matsyasana_-d': 808,
    'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a': 388,
    'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-b': 410,
    'Frog_Pose_or_Bhekasana-a': 886,
    'Garland_Pose_or_Malasana_-a': 334,
    'Gate_Pose_or_Parighasana_-a': 540,
    'Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_-a': 568,
    'Half_Moon_Pose_or_Ardha_Chandrasana_-a': 750,
    'Half_Moon_Pose_or_Ardha_Chandrasana_-b': 688,
    'Happy_Baby_Pose-a': 540,
    'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_-a': 680,
    'Heron_Pose_or_Krounchasana_-a': 658,
    'Intense_Side_Stretch_Pose_or_Parsvottanasana_-a': 592,
    'Legs-Up-the-Wall_Pose_or_Viparita_Karani_-a': 456,
    'Locust_Pose_or_Salabhasana_-a': 458,
    'Locust_Pose_or_Salabhasana_-b': 446,
    'Low_Lunge_pose_or_Anjaneyasana_-a': 848,
    'Low_Lunge_pose_or_Anjaneyasana_-b': 526,
    'Low_Lunge_pose_or_Anjaneyasana_-c': 600,
    'Low_Lunge_pose_or_Anjaneyasana_-d': 684,
    'Noose_Pose_or_Pasasana_-b': 544,
    'Peacock_Pose_or_Mayurasana_-a': 732,
    'Peacock_Pose_or_Mayurasana_-b': 1296,
    'Plank_Pose_or_Kumbhakasana_-a': 398,
    'Plow_Pose_or_Halasana_-a': 742,
    'Plow_Pose_or_Halasana_-b': 720,
    'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a': 484,
    'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b': 514,
    'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_-b': 666,
    'Scale_Pose_or_Tolasana_-a': 914,
    'Scorpion_pose_or_vrischikasana-a': 1400,
    'Scorpion_pose_or_vrischikasana-c': 1082,
    'Shoulder-Pressing_Pose_or_Bhujapidasana_-a': 580,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-a': 746,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-b': 738,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-c': 1162,
    'Side_Plank_Pose_or_Vasisthasana_-a': 500,
    'Side_Plank_Pose_or_Vasisthasana_-b': 602,
    'Side_Plank_Pose_or_Vasisthasana_-c': 888,
    'Side_Plank_Pose_or_Vasisthasana_-d': 500,
    'Side_Plank_Pose_or_Vasisthasana_-e': 554,
    'Sitting_pose_1_(normal)-a': 630,
    'Sitting_pose_1_(normal)-b': 358,
    'Sitting_pose_1_(normal)-c': 440,
    'Split_pose-a': 938,
    'Split_pose-b': 768,
    'Split_pose-c': 462,
    'Staff_Pose_or_Dandasana_-a': 514,
    'Staff_Pose_or_Dandasana_-b': 458,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-a': 456,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-b': 506,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c': 396,
    'Standing_Forward_Bend_pose_or_Uttanasana_-a': 442,
    'Standing_Forward_Bend_pose_or_Uttanasana_-b': 382,
    'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a': 380,
    'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-b': 370,
    'Supported_Headstand_pose_or_Salamba_Sirsasana_-b': 826,
    'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a': 640,
    'Supta_Baddha_Konasana_-b': 754,
    'Supta_Virasana_Vajrasana-c': 966,
    'Tree_Pose_or_Vrksasana_-a': 446,
    'Tree_Pose_or_Vrksasana_-b': 360,
    'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_-a': 612,
    'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_-b': 788,
    'Upward_Plank_Pose_or_Purvottanasana_-a': 592,
    'Upward_Plank_Pose_or_Purvottanasana_-b': 550,
    'viparita_virabhadrasana_or_reverse_warrior_pose-a': 538,
    'viparita_virabhadrasana_or_reverse_warrior_pose-b': 436,
    'Virasana_or_Vajrasana-a': 438,
    'Virasana_or_Vajrasana-c': 428,
    'Warrior_I_Pose_or_Virabhadrasana_I_-a': 518,
    'Warrior_II_Pose_or_Virabhadrasana_II_-a': 446,
    'Warrior_III_Pose_or_Virabhadrasana_III_-a': 630,
    'Warrior_III_Pose_or_Virabhadrasana_III_-b': 400,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-a': 1172,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-b': 682,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-c': 562,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-d': 610,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-a': 872,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-c': 826,
    'Wild_Thing_pose_or_Camatkarasana_-a': 692,
    'Handstand_pose_or_Adho_Mukha_Vrksasana_-a': 622,
    'Scorpion_pose_or_vrischikasana-b': 1946,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-b': 720,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-c': 540,
    'Bound_Angle_Pose_or_Baddha_Konasana_-d': 1090,
    'Bound_Angle_Pose_or_Baddha_Konasana_-e': 788,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a': 420,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-b': 600,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-c': 540,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-d': 550,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-e': 582,
    'Camel_Pose_or_Ustrasana_-a': 684,
    'Camel_Pose_or_Ustrasana_-b': 360,
    'Child_Pose_or_Balasana_-b': 360,
    'Cow_Face_Pose_or_Gomukhasana_-d': 780,
    'Fish_Pose_or_Matsyasana_-b': 850,
    'Frog_Pose_or_Bhekasana-b': 580,
    'Frog_Pose_or_Bhekasana-c': 600,
    'Garland_Pose_or_Malasana_-b': 780,
    'Garland_Pose_or_Malasana_-c': 360,
    'Gate_Pose_or_Parighasana_-b': 480,
    'Heron_Pose_or_Krounchasana_-b': 1020,
    'Heron_Pose_or_Krounchasana_-c': 600,
    'Intense_Side_Stretch_Pose_or_Parsvottanasana_-b': 1080,
    'Locust_Pose_or_Salabhasana_-c': 720,
    'Lord_of_the_Dance_Pose_or_Natarajasana_-a': 480,
    'Lord_of_the_Dance_Pose_or_Natarajasana_-c': 1620,
    'Noose_Pose_or_Pasasana_-a': 780,
    'Rajakapotasana-a': 1536,
    'Rajakapotasana-b': 1500,
    'Rajakapotasana-c': 1260,
    'Rajakapotasana-d': 1500,
    'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_-a': 420,
    'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_-b': 600,
    'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_-a': 900,
    'Bow_Pose_or_Dhanurasana_-a': 660,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c': 360,
    'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_-b': 660,
    'Seated_Forward_Bend_pose_or_Paschimottanasana_-a': 600,
    'Side-Reclining_Leg_Lift_pose_or_Anantasana_-a': 780,
    'Supported_Headstand_pose_or_Salamba_Sirsasana_-a': 754,
    'Supta_Baddha_Konasana_-a': 620,
    'Supta_Virasana_Vajrasana-a': 800,
    'Supta_Virasana_Vajrasana-b': 1100,
    'Supta_Virasana_Vajrasana-d': 780,
    'Tortoise_Pose-a': 960,
    'Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_-a': 1080,
    'Virasana_or_Vajrasana-b': 480,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-b': 840,
    'Wind_Relieving_pose_or_Pawanmuktasana-a': 320,
    'Wind_Relieving_pose_or_Pawanmuktasana-b': 360,
    'Yogic_sleep_pose-a': 1624,
    'Yogic_sleep_pose-b': 1080
}

frame_select_dict_first_session = {
    'Akarna_Dhanurasana-a': 1226,
    'Akarna_Dhanurasana-b': 1126,
    'Akarna_Dhanurasana-c': 730,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-a': 872,
    'Boat_Pose_or_Paripurna_Navasana_-a': 486,
    'Boat_Pose_or_Paripurna_Navasana_-b': 392,
    'Boat_Pose_or_Paripurna_Navasana_-c': 340,
    'Boat_Pose_or_Paripurna_Navasana_-d': 300,
    'Boat_Pose_or_Paripurna_Navasana_-e': 314,
    'Boat_Pose_or_Paripurna_Navasana_-f': 280,
    'Bound_Angle_Pose_or_Baddha_Konasana_-a': 622,
    'Bound_Angle_Pose_or_Baddha_Konasana_-b': 510,
    'Bound_Angle_Pose_or_Baddha_Konasana_-c': 462,
    'Bound_Angle_Pose_or_Baddha_Konasana_-f': 390,
    'Bound_Angle_Pose_or_Baddha_Konasana_-g': 274,
    'Camel_Pose_or_Ustrasana_-c': 430,
    'Camel_Pose_or_Ustrasana_-d': 388,
    'Cat_Cow_Pose_or_Marjaryasana_-a': 326,
    'Cat_Cow_Pose_or_Marjaryasana_-b': 234,
    'Chair_Pose_or_Utkatasana_-a': 266,
    'Chair_Pose_or_Utkatasana_-b': 272,
    'Chair_Pose_or_Utkatasana_-c': 368,
    'Child_Pose_or_Balasana_-a': 468,
    'Cobra_Pose_or_Bhujangasana_-a': 594,
    'Cobra_Pose_or_Bhujangasana_-b': 470,
    'Cobra_Pose_or_Bhujangasana_-c': 444,
    'Cobra_Pose_or_Bhujangasana_-d': 476,
    'Corpse_Pose_or_Savasana_-a': 376,
    'Cow_Face_Pose_or_Gomukhasana_-a': 990,
    'Cow_Face_Pose_or_Gomukhasana_-b': 650,
    'Cow_Face_Pose_or_Gomukhasana_-c': 628,
    'Cow_Face_Pose_or_Gomukhasana_-e': 578,
    'Crane_(Crow)_Pose_or_Bakasana_-a': 848,
    'Crane_(Crow)_Pose_or_Bakasana_-b': 574,
    'Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a': 468,
    'Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a': 416,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a': 432,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-b': 380,
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-c': 340,
    'Eagle_Pose_or_Garudasana_-a': 580,
    'Eight-Angle_Pose_or_Astavakrasana_-a': 686,
    'Extended_Puppy_Pose_or_Uttana_Shishosana_-a': 464,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-a': 590,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-b': 538,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-d': 634,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-e': 510,
    'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-a': 496,
    'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-b': 572,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a': 1216,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-b': 752,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c': 1546,
    'Feathered_Peacock_Pose_or_Pincha_Mayurasana_-d': 766,
    'Firefly_Pose_or_Tittibhasana_-a': 644,
    'Firefly_Pose_or_Tittibhasana_-b': 496,
    'Fish_Pose_or_Matsyasana_-a': 1286,
    'Fish_Pose_or_Matsyasana_-c': 1160,
    'Fish_Pose_or_Matsyasana_-d': 808,
    'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a': 388,
    'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-b': 410,
    'Frog_Pose_or_Bhekasana-a': 886,
    'Garland_Pose_or_Malasana_-a': 334,
    'Gate_Pose_or_Parighasana_-a': 540,
    'Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_-a': 568,
    'Half_Moon_Pose_or_Ardha_Chandrasana_-a': 750,
    'Half_Moon_Pose_or_Ardha_Chandrasana_-b': 688,
    'Happy_Baby_Pose-a': 540,
    'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_-a': 680,
    'Heron_Pose_or_Krounchasana_-a': 658,
    'Intense_Side_Stretch_Pose_or_Parsvottanasana_-a': 592,
    'Legs-Up-the-Wall_Pose_or_Viparita_Karani_-a': 456,
    'Locust_Pose_or_Salabhasana_-a': 458,
    'Locust_Pose_or_Salabhasana_-b': 446,
    'Low_Lunge_pose_or_Anjaneyasana_-a': 848,
    'Low_Lunge_pose_or_Anjaneyasana_-b': 526,
    'Low_Lunge_pose_or_Anjaneyasana_-c': 600,
    'Low_Lunge_pose_or_Anjaneyasana_-d': 684,
    'Noose_Pose_or_Pasasana_-b': 544,
    'Peacock_Pose_or_Mayurasana_-a': 732,
    'Peacock_Pose_or_Mayurasana_-b': 1296,
    'Plank_Pose_or_Kumbhakasana_-a': 398,
    'Plow_Pose_or_Halasana_-a': 742,
    'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a': 484,
    'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b': 514,
    'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_-b': 666,
    'Scale_Pose_or_Tolasana_-a': 914,
    'Scorpion_pose_or_vrischikasana-a': 1400,
    'Scorpion_pose_or_vrischikasana-c': 1082,
    'Shoulder-Pressing_Pose_or_Bhujapidasana_-a': 580,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-a': 746,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-b': 738,
    'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_-c': 1162,
    'Side_Plank_Pose_or_Vasisthasana_-a': 500,
    'Side_Plank_Pose_or_Vasisthasana_-b': 602,
    'Side_Plank_Pose_or_Vasisthasana_-c': 888,
    'Side_Plank_Pose_or_Vasisthasana_-d': 500,
    'Side_Plank_Pose_or_Vasisthasana_-e': 554,
    'Sitting_pose_1_(normal)-a': 630,
    'Sitting_pose_1_(normal)-b': 358,
    'Sitting_pose_1_(normal)-c': 440,
    'Split_pose-a': 938,
    'Split_pose-b': 768,
    'Split_pose-c': 462,
    'Staff_Pose_or_Dandasana_-a': 514,
    'Staff_Pose_or_Dandasana_-b': 458,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-a': 456,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-b': 506,
    'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c': 396,
    'Standing_Forward_Bend_pose_or_Uttanasana_-a': 442,
    'Standing_Forward_Bend_pose_or_Uttanasana_-b': 382,
    'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-a': 380,
    'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_-b': 370,
    'Supported_Headstand_pose_or_Salamba_Sirsasana_-b': 826,
    'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a': 640,
    'Supta_Baddha_Konasana_-b': 754,
    'Supta_Virasana_Vajrasana-c': 966,
    'Tree_Pose_or_Vrksasana_-a': 446,
    'Tree_Pose_or_Vrksasana_-b': 360,
    'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_-a': 612,
    'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_-b': 788,
    'Upward_Plank_Pose_or_Purvottanasana_-a': 592,
    'Upward_Plank_Pose_or_Purvottanasana_-b': 550,
    'viparita_virabhadrasana_or_reverse_warrior_pose-a': 538,
    'viparita_virabhadrasana_or_reverse_warrior_pose-b': 436,
    'Virasana_or_Vajrasana-a': 438,
    'Virasana_or_Vajrasana-c': 428,
    'Warrior_I_Pose_or_Virabhadrasana_I_-a': 518,
    'Warrior_II_Pose_or_Virabhadrasana_II_-a': 446,
    'Warrior_III_Pose_or_Virabhadrasana_III_-a': 630,
    'Warrior_III_Pose_or_Virabhadrasana_III_-b': 400,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-a': 1172,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-b': 682,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-c': 562,
    'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-d': 610,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-a': 872,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-c': 826,
    'Wild_Thing_pose_or_Camatkarasana_-a': 692
}

frame_select_dict_second_session = {
    'Handstand_pose_or_Adho_Mukha_Vrksasana_-a': 622,
    'Scorpion_pose_or_vrischikasana-b': 1946,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-b': 720,
    'Bharadvajas_Twist_pose_or_Bharadvajasana_I_-c': 540,
    'Bound_Angle_Pose_or_Baddha_Konasana_-d': 1090,
    'Bound_Angle_Pose_or_Baddha_Konasana_-e': 788,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-a': 420,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-b': 600,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-c': 540,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-d': 550,
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_-e': 582,
    'Camel_Pose_or_Ustrasana_-a': 684,
    'Camel_Pose_or_Ustrasana_-b': 360,
    'Child_Pose_or_Balasana_-b': 360,
    'Cow_Face_Pose_or_Gomukhasana_-d': 780,
    'Fish_Pose_or_Matsyasana_-b': 850,
    'Frog_Pose_or_Bhekasana-b': 580,
    'Frog_Pose_or_Bhekasana-c': 600,
    'Garland_Pose_or_Malasana_-b': 780,
    'Garland_Pose_or_Malasana_-c': 360,
    'Gate_Pose_or_Parighasana_-b': 480,
    'Heron_Pose_or_Krounchasana_-b': 1020,
    'Heron_Pose_or_Krounchasana_-c': 600,
    'Intense_Side_Stretch_Pose_or_Parsvottanasana_-b': 1080,
    'Locust_Pose_or_Salabhasana_-c': 720,
    'Lord_of_the_Dance_Pose_or_Natarajasana_-a': 480,
    'Lord_of_the_Dance_Pose_or_Natarajasana_-c': 1620,
    'Noose_Pose_or_Pasasana_-a': 780,
    'Plow_Pose_or_Halasana_-b': 720,
    'Rajakapotasana-a': 1536,
    'Rajakapotasana-b': 1500,
    'Rajakapotasana-c': 1260,
    'Rajakapotasana-d': 1500,
    'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_-a': 420,
    'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_-b': 600,
    'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_-a': 900,
    'Bow_Pose_or_Dhanurasana_-a': 660,
    'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c': 360,
    'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_-b': 660,
    'Seated_Forward_Bend_pose_or_Paschimottanasana_-a': 600,
    'Side-Reclining_Leg_Lift_pose_or_Anantasana_-a': 780,
    'Supported_Headstand_pose_or_Salamba_Sirsasana_-a': 754,
    'Supta_Baddha_Konasana_-a': 620,
    'Supta_Virasana_Vajrasana-a': 800,
    'Supta_Virasana_Vajrasana-b': 1100,
    'Supta_Virasana_Vajrasana-d': 780,
    'Tortoise_Pose-a': 960,
    'Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_-a': 1080,
    'Virasana_or_Vajrasana-b': 480,
    'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_-b': 840,
    'Wind_Relieving_pose_or_Pawanmuktasana-a': 320,
    'Wind_Relieving_pose_or_Pawanmuktasana-b': 360,
    'Yogic_sleep_pose-a': 1624,
    'Yogic_sleep_pose-b': 1080
}

# list of mismatch names: vicon_name:ioi_name
name_changes_first_session = {
    'Akarna_Dhanurasana-a': 'Akarna_Dhanurasana-b_1',
    'Akarna_Dhanurasana-b': 'Akarna_Dhanurasana-b_2',
    'Side-Reclining_Leg_Lift_pose_or_Anantasana_-a': 'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_-a',
    'Rajakapotasana-c': 'check_name',
    'Side-Reclining_Leg_Lift_pose_or_Anantasana_-a': 'CoreView2',
    'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a': 'Pose_Dedicated_to_the_Sage_Ko_or_Eka_Pada_Kound_I_and_II-a'

}

# list of rejected names with frame_nums useful in matching IOI with Vicon
reject_names_first_session = []
