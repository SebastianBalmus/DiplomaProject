import type {} from '@mui/material/themeCssVarsAugmentation';
import { createTheme, alpha  } from '@mui/material/styles';

declare module '@mui/material/styles/createPalette' {
  interface ColorRange {
    50: string;
    100: string;
    200: string;
    300: string;
    400: string;
    500: string;
    600: string;
    700: string;
    800: string;
    900: string;
  }

  interface PaletteColor extends ColorRange {}
}

export const brand = {
  50: '#F0F7FF',
  100: '#CEE5FD',
  200: '#9CCCFC',
  300: '#55A6F6',
  400: '#0A66C2',
  500: '#0959AA',
  600: '#064079',
  700: '#033363',
  800: '#02294F',
  900: '#021F3B',
};

export const secondary = {
  50: '#F9F0FF',
  100: '#E9CEFD',
  200: '#D49CFC',
  300: '#B355F6',
  400: '#750AC2',
  500: '#6709AA',
  600: '#490679',
  700: '#3B0363',
  800: '#2F024F',
  900: '#23023B',
};

export const gray = {
  50: '#FBFCFE',
  100: '#EAF0F5',
  200: '#D6E2EB',
  300: '#BFCCD9',
  400: '#94A6B8',
  500: '#5B6B7C',
  600: '#4C5967',
  700: '#364049',
  800: '#131B20',
  900: '#090E10',
};

export const green = {
  50: '#F6FEF6',
  100: '#E3FBE3',
  200: '#C7F7C7',
  300: '#A1E8A1',
  400: '#51BC51',
  500: '#1F7A1F',
  600: '#136C13',
  700: '#0A470A',
  800: '#042F04',
  900: '#021D02',
};

const darkTheme = createTheme({
    palette: {
      mode: 'dark',
      primary: {
        contrastText: brand[100],
        light: brand[200],
        main: brand[300],
        dark: brand[400],
      },
      secondary: {
        light: secondary[400],
        main: secondary[500],
        dark: secondary[900],
      },
      warning: {
        main: '#F7B538',
        dark: '#F79F00',
      },
      error: {
        light: '#D32F2F',
        main: '#D32F2F',
        dark: '#B22A2A'
      },
      success: {
        light: green[400],
        main: green[500],
        dark: green[700],
      },
      grey: {
        50: gray[50],
        100: gray[100],
        200: gray[200],
        300: gray[300],
        400: gray[400],
        500: gray[500],
        600: gray[600],
        700: gray[700],
        800: gray[800],
        900: gray[900],
      },
      background: {
        default: gray[900],
        paper: gray[800],
      },
      text: {
        primary: '#fff',
        secondary: gray[400]
      },
      action: {
        selected: alpha(brand[800], 0.2),
      },
      typography: {
        fontFamily: ['"Inter", "sans-serif"'].join(','),
        h1: {
          fontSize: 60,
          fontWeight: 600,
          lineHeight: 78 / 70,
          letterSpacing: -0.2,
        },
        h2: {
          fontSize: 48,
          fontWeight: 600,
          lineHeight: 1.2,
        },
        h3: {
          fontSize: 42,
          lineHeight: 1.2,
        },
        h4: {
          fontSize: 36,
          fontWeight: 500,
          lineHeight: 1.5,
        },
        h5: {
          fontSize: 20,
          fontWeight: 600,
        },
        h6: {
          fontSize: 18,
        },
        subtitle1: {
          fontSize: 18,
        },
        subtitle2: {
          fontSize: 16,
        },
        body1: {
          fontWeight: 400,
          fontSize: 15,
        },
        body2: {
          fontWeight: 400,
          fontSize: 14,
        },
        caption: {
          fontWeight: 400,
          fontSize: 12,
        },
      },
    },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: gray[900],
            color: gray[100],
          },
        },
      }
    },
  },
);

export default darkTheme;
