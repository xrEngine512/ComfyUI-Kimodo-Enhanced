# Bundled rigged character

- **Model:** Superhero Male, Quaternius Universal Base Characters — Standard.
- **Original asset license:** CC0 1.0 Universal; the source license is preserved in `QUATERNIUS-LICENSE.txt`.
- **Official pack/download:** <https://quaternius.com/packs/universalbasecharacters.html> · <https://quaternius.itch.io/universal-base-characters>
- **Packaged GLB source:** <https://github.com/programasweights/avatar/blob/main/ASSETS.md> and `public/assets/character.glb`. That package documents preserved geometry, normals, skin weights, bone names and reference pose, with its textured materials replaced by matte jade and image payloads omitted. The source repository's MIT license is included as `MIT-LICENSE.txt`.
- **Rig:** one 65-joint skin, T-pose, three skinned triangle primitives; the renderer maps Kimodo body joints to the model's `pelvis`, `spine_01`, `upperarm_l`, and corresponding rig names. Fingers remain in the neutral pose.
- **SHA-256 of the bundled GLB:** `d6f3b64cab629f6ddca9ba063811aeffd3663a8a4e49ade47cd9505dc2bbd70f`

The renderer retargets the selected Kimodo sample to this asset only when `render_style="mesh"` is selected. Other render styles remain joint visualizations.
