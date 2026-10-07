# Local deployment

## Narrow input access

Find the selected stable device and inspect its USB properties:

```bash
udevadm info -q property \
  -n /dev/input/by-id/REPLACE_WITH_YOUR_DEVICE
```

Copy `99-interview-helper.rules.example` to
`/etc/udev/rules.d/99-interview-helper.rules`, replace the example vendor and
product IDs with the inspected values, then reload rules and reconnect the
device:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=input
```

The rule tags the selected device for `uaccess` and invokes the seat-aware ACL
builtin directly because this project installs the rule late as `99-...`. It
grants the active local desktop seat access to only the matched device. Verify
with `getfacl /dev/input/eventN`. Do not make event devices world readable.

If `keyd` remaps the keyboard, it normally grabs the physical keyboard node.
Reading that node will then produce no events even when its permissions look
correct. Use the example rule for `keyd virtual keyboard`; it creates a stable
`/dev/input/by-id/interview-helper-keyd-event-kbd` link and grants access only
to keyd's virtual keyboard output. The app only observes that output and never
calls `grab()`, so keys continue to reach the desktop.

Bluetooth devices commonly lack `/dev/input/by-id` links. Inspect the device's
stable identity and parent attributes with:

```bash
udevadm info --attribute-walk --name=/dev/input/eventN
```

Use the Bluetooth template in the example rule, including the paired device's
`uniq` value, to create a stable link and a narrow active-seat ACL. Avoid saving
the reconnect-dependent `/dev/input/eventN` path in the app.

## User service

Copy `interview-helper.service.example` to
`~/.config/systemd/user/interview-helper.service`, replace the device path and
event code, and adjust `WorkingDirectory` and `ExecStart` to your checkout and
virtual environment (`%h` expands to your home directory), then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now interview-helper.service
journalctl --user -u interview-helper.service -f
```

The service intentionally runs as the desktop user and has no root privileges.
Its stdout contains transcripts, so decide whether journal retention is
acceptable before enabling it. For stricter privacy, run it interactively or
replace stdout delivery with an ephemeral consumer.
