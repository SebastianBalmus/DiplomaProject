import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import VoiceChatIcon from '@mui/icons-material/VoiceChat';


function ResponsiveAppBar() {
  return (
    <AppBar 
      position="static"
      color="primary"
    >
      <Container maxWidth="xl">
        <Toolbar
          disableGutters
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
        <VoiceChatIcon sx={{ display: 'flex', mr: 1 }} />
        <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              display: 'flex',
            }}
          >
            Text to 
            <Typography
              variant="h6"
              noWrap
              sx={{
                fontWeight: 700,
                color: 'primary.main',
              }}
            >
              &nbsp;Speech
            </Typography>
          </Typography>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
export default ResponsiveAppBar;
