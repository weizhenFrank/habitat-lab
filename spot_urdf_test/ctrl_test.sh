#/bin/bash


# export PYTHONPATH=$PYTHONPATH:/nethome/mrudolph8/Documents/haburdf/habitat-sim/
cd /nethome/mrudolph8/Documents/haburdf/habitat-sim

ln -s /srv/share3/mrudolph8/data/data/datasets /nethome/mrudolph8/Documents/haburdf/habitat-sim/data/
ln -s /srv/share3/mrudolph8/data/data/scene_datasets /nethome/mrudolph8/Documents/haburdf/habitat-sim/data/
ln -s /srv/share3/mrudolph8/data/data/URDF_demo_assets /nethome/mrudolph8/Documents/haburdf/habitat-sim/data/


python ../spot_urdf_test/raibert_controller_test.py